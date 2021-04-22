#!/usr/bin/env python

"""
Classification of cells into cell types/states.
"""


from typing import Callable, List, Tuple
import sys, re, json
from argparse import ArgumentParser

import anndata
import scanpy as sc  # type: ignore[import]
from numpy_groupies import aggregate
from joblib import parallel_backend
import pingouin as pg

from imc.graphics import get_grid_dims, rasterize_scanpy, add_centroids
from imc.utils import align_channels_by_name
from imc.types import DataFrame

from seaborn_extensions import swarmboxenplot, clustermap

from src.config import *

# from src.operations import (
#     plot_umap,
#     plot_umap_with_labeled_clusters,
#     plot_cluster_heatmaps,
#     plot_cluster_illustrations,
#     plot_cluster_heatmaps_with_labeled_clusters,
# )
# from src.utils import z_score_by_column


cli = None


def main(cli=None) -> int:
    parser = get_parser()
    args = parser.parse_args(cli)

    # Phenotype all cells
    phenotyping(args, remove_batch_method="combat.bbknn")

    # increased resolution of CD4 T-cells
    phenotyping_t_cells(args, suffix=".cd4_tcell_clustered")

    # Set cell type names
    h5ad_f = Path(
        results_dir / "phenotyping" / prj.name
        + ".filter_solidity.log1p.combat.bbknn.h5ad"
    )
    a = sc.read(h5ad_f)

    # Add a cell_type label
    if "cell_type" not in a.obs.columns:
        ct_f = metadata_dir / "cluster_labels.json"
        cluster_labels = json.load(open(ct_f, "r"))["1.0"]
        cluster_labels = {int(k): v for k, v in cluster_labels.items()}
        a.obs["cell_type"] = a.obs["cluster_1.0"].replace(cluster_labels)

        # # label Tregs (cluster 3 of more refined CD4 clustering)
        h5ad_rf = h5ad_f.replace_(".h5ad", ".cd4_tcell_clustered" + ".h5ad")
        a2 = sc.read(h5ad_rf)
        cells = a2.obs.index[a2.obs["rcluster_0.5"] == 3]
        l = "15 - T-Regs (CD3, CD4, CD45, CD45RO, FoxP3)"
        a.obs.loc[cells, "cell_type"] = l

        # #
        t = a.obs["cell_type"].str.contains("CD8 T cells")
        t_expr = a[t, "CD8a(Dy162)"].X.squeeze()
        a2 = a[t, :]
        # plt.scatter(*a2.obsm['X_umap'].T, s=2, alpha=0.5)
        # plt.axhline(15, color='grey', linestyle='--')
        # plt.axvline(6.85, color='grey', linestyle='--')
        u = a.obsm["X_umap"]
        l = "16 - ? ()"
        a.obs.loc[t & ((u[:, 0] < 6.85) | (u[:, 1] < 15)), "cell_type"] = l

        a.obs["cell_type_broad"] = (
            a.obs["cell_type"]
            .str.extract(r"\d+ - (.*) \(.*")[0]
            .replace("?", "Unkown")
        )
        o = a.obs["cell_type_broad"].value_counts().index
        o = {c: f"{str(i).zfill(2)} - {c}" for i, c in enumerate(o, 1)}
        a.obs["cell_type_broad"] = a.obs["cell_type_broad"].replace(o)

        # save h5ad
        sc.write(h5ad_f, a)

    # # replot with cell type labels
    p = results_dir / "phenotyping" / prj.name
    fig = sc.pl.umap(a, color="cell_type", show=False).figure
    rasterize_scanpy(fig)
    a.obs["cell_type_i"] = pd.Categorical(
        a.obs["cell_type"].str.extract(r"(\d+) - .*")[0]
    )
    add_centroids(a, column="cell_type_i", ax=fig.axes[0], algo="umap")
    fig.savefig(p + ".umap.cell_type.svg", **figkws)

    # # replot heatmap
    chs = state_channels + ["DNA", "area", "eccentricity"]
    x = a.to_df().join(a.obs[chs]).groupby(a.obs["cell_type"]).mean()
    cells = a.obs["cell_type"].value_counts().rename("Cell count")
    kws = dict(row_colors=cells, figsize=(9, 5))
    fig = clustermap(x[a.var.index], config="abs", **kws).fig
    fig.savefig(p + ".heatmap.cell_type.abs.svg", **figkws)
    fig = clustermap(x, config="z", **kws).fig
    fig.savefig(p + ".heatmap.cell_type.z.svg", **figkws)

    # # illustrate cell types spatially
    chs = [
        "ColtypeI(Tm169)",
        "KRT5(Dy163)",
        "PanKeratinC11(Nd148)",
        "GATA3(Eu153)",
        "CD3(Er170)",
        "CD4(Gd156)",
        "FoxP3(Gd155)",
        "CD8a(Dy162)",
        "CD68(Tb159)",
        "CD20(Dy161)",
    ]
    illustrate_phenotypes(
        a, channels=chs, labels=["cell_type", "cell_type_broad"]
    )

    # Make cluster and cell type counts per sample and ROI
    _ = generate_count_matrices(a, var="cell_type", overwrite=True)

    # See distribution of abundance between samples with different attributes
    differential_abundance()

    # Regression
    regression()

    return 0


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false"
    )
    parser.add_argument(
        "--resolutions", default=[0.5, 1.0, 1.5, 2.0], nargs="+"
    )
    parser.add_argument(
        "--algos", default=["umap", "diffmap", "pymde", "draw_graph"], nargs="+"
    )
    return parser


def quantify_subcellular_localization() -> DataFrame:
    import parmap
    from skimage.morphology import binary_dilation, binary_erosion, disk

    def quantify(roi) -> DataFrame:
        nucl_mask = roi.nuclei_mask.astype(int)
        cell_mask = roi.cell_mask.astype(int)
        cyto_mask = binary_erosion(cell_mask, disk(2))
        cyto_mask[nucl_mask != 0] = 0
        membr_mask = cell_mask.copy()
        o = (nucl_mask > 0) | (cyto_mask > 0)
        membr_mask[o] = 0
        stack = roi.stack
        n = stack[:, nucl_mask > 0].mean(1)
        c = stack[:, cyto_mask.astype(int) > 0].mean(1)
        m = stack[:, membr_mask > 0].mean(1)
        e = stack[:, cell_mask == 0].mean(1)
        return pd.DataFrame(
            [n, c, m, e],
            index=["nuclear", "cytoplasmatic", "membranar", "extracellular"],
            columns=roi.channel_labels,
        ).T
        # # to visualize:
        # p = np.zeros(roi.shape[1:], dtype=int)
        # p[nucl_mask > 0] = 1
        # p[cyto_mask.astype(int) > 0] = 2
        # p[membr_mask > 0] = 3
        # p[cell_mask == 0] = 4
        # fig, ax = plt.subplots()
        # ax.imshow(p, cmap='tab10')

    _quants = parmap.map(quantify, prj.rois, pm_pbar=True)
    quant = pd.concat(
        [q.assign(roi=roi.name) for q, roi in zip(_quants, prj.rois)]
    )
    quant["nuclear_cytolasmatic_ratio"] = np.log2(
        quant["nuclear"] / quant["cytoplasmatic"]
    )
    quant["cytoplasmatic_membranar_ratio"] = np.log2(
        quant["cytoplasmatic"] / quant["membranar"]
    )
    quant["cellular_extracellular_ratio"] = np.log2(
        quant[["nuclear", "cytoplasmatic", "membranar"]].mean(1)
        / quant["extracellular"]
    )
    quant.to_csv(results_dir / "subcellular_quantification.whole_image.csv")

    p = quant.groupby(level=0).mean()
    p = p.loc[~p.index.str.contains("EMPTY")]
    s = quant.groupby(level=0).std()
    s = s.loc[~s.index.str.contains("EMPTY")]

    conds = [
        (
            "cytoplasmatic_membranar_ratio",
            "nuclear_cytolasmatic_ratio",
            "cellular_extracellular_ratio",
        ),
        (
            "cellular_extracellular_ratio",
            "nuclear_cytolasmatic_ratio",
            "cytoplasmatic_membranar_ratio",
        ),
    ]
    fig, axes = plt.subplots(1, len(conds), figsize=(10 * len(conds), 10))
    for ax, (x, y, z) in zip(axes, conds):
        ax.errorbar(
            x=p[x],
            y=p[y],
            yerr=s[y],
            xerr=s[x],
            linestyle="None",
            marker="^",
            ecolor="grey",
            alpha=0.25,
        )
        ax.scatter(data=p, x=x, y=y, c=z, alpha=0.5)
        ax.set(xlabel=x, ylabel=y)
        for m in p.index:
            ax.text(p.loc[m, x], p.loc[m, y], ha="center", va="center", s=m)
    for ax in axes:
        ax.axhline(0, color="grey", linestyle="--")
        ax.axvline(0, color="grey", linestyle="--")
    fig.savefig(
        results_dir / "subcellular_quantification.whole_image.svg", **figkws
    )


def phenotyping(
    args,
    suffix="",
    filter_cells: bool = True,
    filter_cells_method: str = "solidity",
    log_transform: bool = True,
    z_scale: bool = False,
    z_scale_per: str = "sample",
    z_scale_cap: int = 3,
    remove_batch: bool = True,
    remove_batch_method: str = "combat.harmony",
) -> None:
    """
    Cluster cells into phenotype groups.
    """
    output_dir = results_dir / "phenotyping"
    output_dir.mkdir()

    # Quantify intensity of channels
    quant_f = output_dir / f"{prj.name}.quantification.pq"
    quant_ff = quant_f.replace_(".pq", ".filtered.pq")
    proc_suffix = ""
    if filter_cells:
        proc_suffix += f".filter_{filter_cells_method}"
    if log_transform:
        proc_suffix += f".log1p"
    if z_scale:
        proc_suffix += f".z_scaled_per_{z_scale_per}"
    if remove_batch:
        proc_suffix += f".{remove_batch_method}"
    h5ad_f = output_dir / f"{prj.name}{suffix}{proc_suffix}.h5ad"

    if not quant_ff.exists() or args.overwrite:
        if not quant_f.exists() or args.overwrite:
            prj.quantify_cells()
            prj.quantification.to_parquet(quant_f)
        quant = pd.read_parquet(quant_f)

        # reduce DNA chanels to one
        dna_cols = quant.columns[quant.columns.str.contains(r"DNA\d")]
        quant["DNA"] = quant[dna_cols].mean(1)
        quant = quant.drop(dna_cols, axis=1)

        if include_channels is not None:
            inc = [
                "DNA",
                "area",
                "perimeter",
                "major_axis_length",
                "eccentricity",
                "solidity",
                "roi",
                "sample",
            ]
            quant = quant.reindex(columns=include_channels + inc)
        if exclude_channels is not None:
            exc = quant.isnull().any() | quant.columns.isin(exclude_channels)
            quant = quant.drop(exc[exc].index, axis=1)

        # filter out cells
        if filter_cells:
            # # GMM
            # exclude = _filter_out_cells(
            #     quant, plot=True, output_prefix=output_dir / prj.name + "."
            # )
            # or, less stringent:
            exclude = quant["solidity"] == 1

            p = (exclude).sum() / quant.shape[0] * 100
            tqdm.write(f"Filtering out {exclude.sum()} cells ({p:.2f} %)")

        quant = quant.loc[~exclude, :]
        quant.to_parquet(quant_ff)

    if not h5ad_f.exists() or args.overwrite:
        # Prepare data
        quant = pd.read_parquet(quant_ff)
        quant.index.name = "obj_id"

        # Drop unwanted channels and redundant morphological feaatures
        exc = "|".join(channels_exclude_strings)
        q = (
            quant[
                quant.columns.to_series().loc[lambda x: ~x.str.contains(exc)]
            ].reset_index()
            # .drop(tech_channels, axis=1)
        )
        a = anndata.AnnData(
            q.drop(id_cols + tech_channels + state_channels, axis=1),
            obs=q[id_cols + tech_channels + state_channels],
        )

        # add clinical annotation
        a.obs = (
            a.obs.reset_index()
            .merge(
                roi_attributes.reset_index(),
                on=["sample", "roi"],
                how="left",
            )
            .set_index("index")
            .loc[a.obs.index]
        )

        a.raw = a
        sc.write(h5ad_f.replace_(".combat", ".raw"), a)

        a = sc.read(h5ad_f.replace_(".combat", ".raw"))

        # preprocess
        if log_transform:
            sc.pp.log1p(a)

        # per image Z-score
        if z_scale:
            _ads = list()
            objs = getattr(prj, z_scale_per + "s")
            for obj in objs:
                a2 = a[a.obs[z_scale_per] == obj.name, :]
                sc.pp.scale(a2, max_value=z_scale_cap)
                a2.X[a2.X < -z_scale_cap] = -z_scale_cap
                _ads.append(a2)
            a = anndata.concat(_ads)

        sc.pp.scale(a)
        if remove_batch:
            if "combat" in remove_batch_method:
                sc.pp.combat(a, "sample")
                sc.pp.scale(a)

        # res dim
        sc.pp.pca(a)
        if remove_batch:
            if "bbknn" in remove_batch_method:
                sc.external.pp.bbknn(a, batch_key="sample")
            if "harmony" in remove_batch_method:
                sc.external.pp.harmony_integrate(a, "sample")
                a.obsm["X_pca"] = a.obsm["X_pca_harmony"]
        else:
            sc.pp.neighbors(a)
        if "umap" in args.algos:
            sc.tl.umap(a, gamma=25)
        if "diffmap" in args.algos:
            sc.tl.diffmap(a)

        if "pymde" in args.algos:
            import pymde

            a.obsm["X_pymde"] = (
                pymde.preserve_neighbors(a.X, embedding_dim=2).embed().numpy()
            )
            a.obsm["X_pymde2"] = (
                pymde.preserve_neighbors(
                    a.X,
                    embedding_dim=2,
                    attractive_penalty=pymde.penalties.Quadratic,
                    repulsive_penalty=None,
                )
                .embed()
                .numpy()
            )

        # cluster
        # # using Leiden
        for res in tqdm(args.resolutions):
            sc.tl.leiden(a, resolution=res, key_added=f"cluster_{res}")
            a.obs[f"cluster_{res}"] = pd.Categorical(
                a.obs[f"cluster_{res}"].astype(int) + 1
            )
            sc.write(h5ad_f, a)

        # # using PARC
        from parc import PARC

        p = PARC(a.X, neighbor_graph=a.obsp["connectivities"], random_seed=42)
        p.run_PARC()
        a.obs["cluster_parc"] = [str(i) for i in p.labels]

        sc.write(h5ad_f, a)

    a = sc.read(h5ad_f)
    a = a[a.obs.sample(frac=1).index]

    # Plot projections
    vmin = None
    vmax = (
        [None]
        + np.percentile(a.raw[:, a.var.index].X, 95, axis=0).tolist()
        + np.percentile(
            a.obs[tech_channels + state_channels], 95, axis=0
        ).tolist()
        + [None] * len(attributes)
        + ([None] * len(args.resolutions))
    )
    color = (
        ["sample"]
        + a.var.index.tolist()
        + tech_channels
        + state_channels
        + attributes
        + [f"cluster_{res}" for res in args.resolutions]
    )
    for algo in tqdm(args.algos):
        f = output_dir / f"{prj.name}.{algo}{suffix}{proc_suffix}.pdf"
        projf = get_scanpy_func(algo)
        axes = projf(
            a,
            color=color,
            show=False,
            vmin=vmin,
            vmax=vmax,
            use_raw=True,
        )
        fig = axes[0].figure
        for ax, res in zip(axes[-len(args.resolutions) :], args.resolutions):
            add_centroids(a, res=res, ax=ax, algo=algo)
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)

        # Plot ROIs separately
        f = (
            output_dir
            / f"{prj.name}.{algo}{suffix}{proc_suffix}.sample_roi.pdf"
        )
        projf = getattr(sc.pl, algo)
        fig = projf(a, color=["sample", "roi"], show=False)[0].figure
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)
        plt.close("all")

    # Plot average phenotypes
    for res in tqdm(args.resolutions):
        df = a.to_df().join(a.obs).drop(id_cols, 1)
        cluster_means = df.groupby(a.obs[f"cluster_{res}"].values).mean()
        cell_counts = (
            a.obs[f"cluster_{res}"].value_counts().rename("Cells per cluster")
        )

        cell_percs = ((cell_counts / cell_counts.sum()) * 100).rename(
            "Cells (%)"
        )

        output_prefix = (
            output_dir
            / f"{prj.name}.cluster_means.{res}_res{suffix}{proc_suffix}."
        )
        kws = dict(
            row_colors=cell_percs.to_frame().join(cell_counts),
            figsize=(10, 6 * res),
        )
        grid = clustermap(cluster_means, **kws)
        grid.savefig(output_prefix + "abs.svg")
        grid = clustermap(cluster_means, **kws, config="z")
        grid.savefig(output_prefix + "zscore.svg")
        grid = clustermap(cluster_means, **kws, config="z", row_cluster=False)
        grid.savefig(output_prefix + "zscore.sorted.svg")
        plt.close("all")


def phenotyping_t_cells(args, suffix=".cd4_tcell_clustered") -> None:
    output_dir = results_dir / "phenotyping"
    output_dir.mkdir()

    # Take only epithelial cells
    h5ad_f = Path(
        "results/phenotyping/utuc-imc.filter_solidity.log1p.combat.bbknn.h5ad"
    )
    h5ad_rf = h5ad_f.replace_(".h5ad", suffix + ".h5ad")
    a = sc.read(h5ad_f)
    cluster_labels = json.load(open(metadata_dir / "cluster_labels.json", "r"))[
        "1.0"
    ]
    clusters = [int(k) for k, v in cluster_labels.items() if "CD4 T cells" in v]
    a2 = a[a.obs["cluster_1.0"].isin(clusters)]

    # Plot original UMAP for epithelial cells only
    f = output_dir / f"{prj.name}{suffix}.orignal_umap_space.pdf"
    fig = sc.pl.umap(a2, show=False, s=2, alpha=0.25).figure
    xm, ym = np.absolute(a.obsm["X_umap"]).max(0)
    fig.axes[0].set(xlim=(-xm, xm), ylim=(-ym, ym))
    rasterize_scanpy(fig)
    fig.savefig(f, **figkws)

    sc.tl.umap(a2, gamma=25)
    if "diffmap" in args.algos:
        sc.tl.diffmap(a2)

    if "pymde" in args.algos:
        import pymde

        a2.obsm["X_pymde"] = (
            pymde.preserve_neighbors(a2.X, embedding_dim=2).embed().numpy()
        )
        a2.obsm["X_pymde2"] = (
            pymde.preserve_neighbors(
                a2.X,
                embedding_dim=2,
                attractive_penalty=pymde.penalties.Quadratic,
                repulsive_penalty=None,
            )
            .embed()
            .numpy()
        )

    if "draw_graph" in args.algos:
        sc.tl.draw_graph(a2)

    # cluster
    for res in tqdm(args.resolutions):
        sc.tl.leiden(a2, resolution=res, key_added=f"rcluster_{res}")
        a2.obs[f"rcluster_{res}"] = pd.Categorical(
            a2.obs[f"rcluster_{res}"].astype(int) + 1
        )

    # Save
    for x in list(a2.uns.keys()):
        if x.endswith("_colors"):
            del a2.uns[x]
    sc.write(h5ad_rf, a2)

    # Plot
    a2 = sc.read(h5ad_rf)
    a2 = a2[a2.obs.sample(frac=1).index]

    # Plot projections
    vmin = None
    vmax = (
        [None]
        + np.percentile(a2.raw[:, a2.var.index].X, 95, axis=0).tolist()
        + np.percentile(
            a2.obs[tech_channels + state_channels], 95, axis=0
        ).tolist()
        + ([None] * len(args.resolutions))
    )
    color = (
        ["sample"]
        + a2.var.index.tolist()
        + tech_channels
        + state_channels
        + [f"rcluster_{res}" for res in args.resolutions]
    )
    for algo in tqdm(args.algos):
        f = output_dir / f"{prj.name}.{algo}{suffix}.pdf"
        projf = get_scanpy_func(algo)
        fig = projf(
            a2,
            color=color,
            show=False,
            vmin=vmin,
            vmax=vmax,
            use_raw=True,
        )[0].figure
        for ax, res in zip(
            fig.axes[-(len(args.resolutions)) :],
            args.resolutions,
        ):
            _algo = algo + "_fa" if algo == "draw_graph" else algo
            add_centroids(a2, column=f"rcluster_{res}", ax=ax, algo=_algo)
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)

        # Plot ROIs separately
        f = output_dir / f"{prj.name}.{algo}{suffix}.sample_roi.pdf"
        projf = getattr(sc.pl, algo)
        fig = projf(a2, color=["sample", "roi"], show=False)[0].figure
        rasterize_scanpy(fig)
        fig.savefig(f, **figkws)
        plt.close("all")

    # Plot average phenotypes
    for res in tqdm(args.resolutions):
        df = a2.to_df()
        cluster_means = df.groupby(a2.obs[f"rcluster_{res}"].values).mean()
        cell_counts = (
            a2.obs[f"rcluster_{res}"].value_counts().rename("Cells per cluster")
        )
        cell_percs = ((cell_counts / cell_counts.sum()) * 100).rename(
            "Cells (%)"
        )

        output_prefix = (
            output_dir / f"{prj.name}.cluster_means.{res}_res{suffix}."
        )
        h = 6 * max(res, 1)
        kws = dict(
            row_colors=cell_percs.to_frame().join(cell_counts),
            figsize=(10, h),
        )
        grid = clustermap(cluster_means, **kws)
        grid.savefig(output_prefix + "abs.svg")
        grid = clustermap(cluster_means, **kws, config="z")
        grid.savefig(output_prefix + "zscore.svg")
        grid = clustermap(cluster_means, **kws, config="z", row_cluster=False)
        grid.savefig(output_prefix + "zscore.sorted.svg")
        plt.close("all")


def illustrate_phenotypes(
    a: anndata.AnnData,
    channels: List[str],
    labels: List[str],
    overwrite: bool = False,
    suffix: str = "",
    remove_files: bool = False,
) -> None:
    import os
    import parmap

    # Plot one just to get it started
    roi = prj.rois[0]
    _plot_roi(
        roi,
        channels=channels[:3],
        clusters=labels,
        overwrite=overwrite,
        output_suffix=suffix,
        remove_cluster_str="Unkown",
        file_type="png",
    )
    # Now plot all
    parmap.map(
        _plot_roi,
        prj.rois,
        pm_pbar=True,
        channels=channels,
        clusters=labels,
        overwrite=overwrite,
        output_suffix=suffix,
        remove_cluster_str="Unkown",
        file_type="pdf",
    )

    out_dir = results_dir / "illustration"
    out_dir.mkdir()
    files = sorted(out_dir.glob(f"*.illustration{suffix}.pdf"))
    out_file = out_dir / "phenotype_illustrations.pdf"
    cmd = f"pdftk {' '.join(map(str, files))} cat output {out_file}"
    os.system(cmd)

    if remove_files:
        for file in files:
            file.unlink(missing_ok=True)


def _plot_roi(
    roi,
    channels=None,
    clusters=None,
    overwrite: bool = False,
    output_suffix="",
    position=None,
    file_type="png",
    return_fig=False,
    remove_cluster_str: str = None,
) -> None:
    if channels is None:
        channels = a.var.index
    (results_dir / "illustration").mkdir()
    out_f = (
        results_dir / "illustration" / roi.name
        + f".illustration{output_suffix}.{file_type}"
    )
    if out_f.exists() and not overwrite:
        return

    ar = roi.shape[1] / roi.shape[2]
    n, m = 2, max(len(channels), len(clusters))
    fig, axes = plt.subplots(
        n,
        m,
        figsize=(m * 4, n * 4 * ar),
        gridspec_kw=dict(wspace=0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    roi.plot_channels(channels, axes=axes[0])
    for i, (ax, cluster) in enumerate(zip(axes[1, ::2], clusters)):
        c = (
            a.obs.set_index(["sample", "roi", "obj_id"])[cluster]
            .rename("cluster")
            .astype(str)
        )
        if remove_cluster_str is not None:
            c = c[~c.str.contains(remove_cluster_str)]
        if all(
            [
                isinstance(x, (int, float, np.int64, np.float64))
                for x in a.obs[cluster]
            ]
        ):
            c = c + " - " + c
        roi.plot_cell_types(c.loc[roi.sample.name, roi.name, :], ax=ax)
        ax.set(title=cluster)
    for ax in axes.flat:
        ax.axis("off")

    fig.savefig(
        out_f,
        dpi=300,
        bbox_inches="tight",
    )
    if return_fig:
        return fig
    plt.close(fig)


def get_scanpy_func(algo: str) -> Callable:
    if algo != "pymde":
        return getattr(sc.pl, algo)
    return partial(sc.pl.scatter, basis="pymde")

    #

    #

    #

    #


def generate_count_matrices(
    a: anndata.AnnData = None,
    var: str = "cell_type",
    exclude_pattern: str = r"\?",
    overwrite: bool = False,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    # Let's generate some dataframes that will be useful later:
    # # counts of cells per image, per cluster
    if (not roi_counts_file.exists()) or overwrite:
        roi_counts = (
            a.obs[["roi", var]]
            .assign(count=1)
            .pivot_table(
                index="roi",
                columns=var,
                values="count",
                aggfunc=sum,
                fill_value=0,
            )
        )
        roi_counts.columns = roi_counts.columns.astype(str)
        if exclude_pattern is not None:
            roi_counts = roi_counts.loc[
                :,
                (~roi_counts.columns.str.contains(exclude_pattern)),
            ]
        roi_counts.to_parquet(roi_counts_file)
    roi_counts = pd.read_parquet(roi_counts_file)

    # # Area per image
    if (not roi_areas_file.exists()) or overwrite:
        roi_areas = pd.Series(
            {r.name: r.area for r in prj.rois}, name="area"
        ).rename_axis("roi")
        roi_areas.to_csv(roi_areas_file)
    roi_areas = pd.read_csv(roi_areas_file, index_col=0, squeeze=True)

    # # counts of cells per sample, per cluster
    if (not sample_counts_file.exists()) or overwrite:
        sample_counts = (
            a.obs[["sample", var]]
            .assign(count=1)
            .pivot_table(
                index="sample",
                columns=var,
                values="count",
                aggfunc=sum,
                fill_value=0,
            )
        )
        sample_counts.columns = sample_counts.columns.astype(str)
        if exclude_pattern is not None:
            sample_counts = sample_counts.loc[
                :, ~sample_counts.columns.str.contains(exclude_pattern)
            ]
        sample_counts.to_parquet(sample_counts_file)
    sample_counts = pd.read_parquet(sample_counts_file)

    # # Area per sample
    if (not sample_areas_file.exists()) or overwrite:
        sample_areas = pd.Series(
            {s.name: sum([r.area for r in s]) for s in prj}, name="area"
        )
        sample_areas.to_csv(sample_areas_file)
    sample_areas = pd.read_csv(sample_areas_file, index_col=0, squeeze=True)

    return (roi_counts, roi_areas, sample_counts, sample_areas)


def differential_abundance() -> None:
    """
    Plot fraction of cells per ROI/Sample and grouped by disease/phenotype.
    """
    output_dir = results_dir / "abundance"
    output_dir.mkdir()

    (
        roi_counts,
        roi_areas,
        sample_counts,
        sample_areas,
    ) = generate_count_matrices()

    # # Heatmaps
    for grouping, df, area, attrs in [
        ("sample", sample_counts, sample_areas, sample_attributes),
        ("roi", roi_counts, roi_areas, roi_attributes),
    ]:
        output_prefix = (
            output_dir
            / f"{prj.name}.clustering.cell_type_abundance.cells_per_{grouping}."
        )
        # groupby meta clusters
        df2 = df.copy()
        df2.columns = df2.columns.str.extract(r"\d+ - (.*) \(")[0]
        df2 = df2.T.groupby(level=0).sum().T.rename_axis(columns="Cluster")

        # # percentage
        dfp = (df.T / df.sum(1)).T * 100
        df2p = (df2.T / df2.sum(1)).T * 100

        # # per mm2
        dfarea = (df.T / area).T * 1e6
        df2area = (df2.T / area).T * 1e6

        kws = dict(
            config="abs",
            xticklabels=True,
            row_colors=attrs,
            cbar_kws=dict(label="Cells (%)"),
        )
        grid = clustermap(dfp, **kws)
        grid.savefig(output_prefix + f"percentage.svg", **figkws)
        grid = clustermap(df2p, **kws)
        grid.savefig(output_prefix + f"percentage.meta.svg", **figkws)

        kws = dict(
            config="abs",
            xticklabels=True,
            row_colors=attrs,
            cbar_kws=dict(label="Cells per mm2"),
        )
        grid = clustermap(dfarea, **kws)
        grid.savefig(output_prefix + f"per_mm2.svg", **figkws)
        grid = clustermap(df2area, **kws)
        grid.savefig(output_prefix + f"per_mm2.meta.svg", **figkws)
        plt.close("all")

    # # Swarmboxenplots
    _stats = list()
    for grouping, df, area, attrs, in [
        ("sample", sample_counts, sample_areas, sample_attributes),
        ("roi", roi_counts, roi_areas, roi_attributes),
    ]:
        output_prefix = (
            output_dir
            / f"{prj.name}.clustering.cell_type_abundance.cells_per_{grouping}."
        )
        # groupby meta clusters
        df2 = df.copy()
        df2.columns = df2.columns.str.extract(r"\d+ - (.*) \(")[0]
        df2 = df2.T.groupby(level=0).sum().T.rename_axis(columns="Cluster")

        for dtype, dfp, dfarea in [
            ("cluster", (df.T / df.sum(1)).T * 100, (df.T / area).T * 1e6),
            ("cell_type", (df2.T / df2.sum(1)).T * 100, (df2.T / area).T * 1e6),
        ]:
            for p, measure in [(dfp, "percentage"), (dfarea, "mm2")]:
                for var in attributes:
                    fig, stats = swarmboxenplot(
                        data=p.join(attrs[[var]]),
                        x=var,
                        y=p.columns,
                        plot_kws=dict(palette=colors[var]),
                    )
                    _stats.append(
                        stats.assign(
                            grouping=grouping, dtype=dtype, measure=measure
                        )
                    )
                    var = var.replace("/", "-").replace(" ", "_")
                    fig.savefig(
                        output_prefix
                        + f"{dtype}.{measure}.{var}.swarmboxenplot.svg",
                        **figkws,
                    )
                    plt.close(fig)
    stats = pd.concat(_stats)
    stats.to_csv(
        output_dir / f"{prj.name}.clustering.cell_type_abundance.stats.csv",
        index=False,
    )

    # Save for supplement
    stats.to_excel("manuscript/Supplementary Table 3.xlsx", index=False)


def regression() -> None:
    """
    Plot fraction of cells per ROI/Sample and grouped by disease/phenotype.
    """
    import matplotlib
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    output_dir = results_dir / "abundance"
    output_dir.mkdir()

    (
        roi_counts,
        roi_areas,
        sample_counts,
        sample_areas,
    ) = generate_count_matrices()

    agg_sample_counts = (
        sample_counts.T.groupby(
            sample_counts.columns.str.extract(r"\d+ - (.*) \(.*")[0].values
        )
        .mean()
        .T
    )
    y = (agg_sample_counts.T / agg_sample_counts.sum(1)).T * 100
    x = sample_attributes.reindex(y.index)

    # clean up names
    y.columns = (
        y.columns.str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    x.columns = (
        x.columns.str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )

    # preparations
    d = x.join(y + 1).dropna()

    _res = list()
    for var in y.columns:
        r = smf.glm(
            f"{var} ~ {' + '.join(x.columns)}",
            data=d,
            family=sm.families.Gamma(sm.families.links.log()),
        ).fit()
        _res.append(r.summary2().tables[1].assign(cell_type=var))
    res = pd.concat(_res).drop("Intercept").rename_axis(index="variable")
    res.to_csv(output_dir / "attribute_regression.csv")

    c = res.pivot_table(index="cell_type", columns="variable", values="Coef.")
    p = res.pivot_table(index="cell_type", columns="variable", values="P>|z|")
    p = (p < 0.05).astype(int)
    grid = clustermap(
        c,
        center=0,
        cmap="RdBu_r",
        cbar_kws=dict(label=r"Coefficient ($\beta$)"),
        figsize=(3, 5),
        annot=p,
    )
    for child in grid.ax_heatmap.get_children():
        if isinstance(child, matplotlib.text.Text):
            if child.get_text() == "0":
                child.set_visible(False)
                child.remove()
            elif child.get_text() == "1":
                child.set_text("*")
    grid.savefig(output_dir / "attribute_regression.svg", **figkws)

    # See if there's any change in subcellular localization ratios
    quant = pd.read_csv(
        results_dir / "subcellular_quantification.whole_image.csv", index_col=0
    ).drop(exclude_channels)
    quant.loc[:, "sample"] = quant["roi"].str.extract(r"(.*)-\d+")[0]
    y = quant.reset_index().groupby(["sample", "channel"]).mean()

    # preparations
    d = x.join(y).dropna()

    _res = list()
    for var in y.columns[y.columns.str.contains("_ratio")]:
        for channel in y.index.levels[1]:
            r = smf.glm(
                f"{var} ~ {' + '.join(x.columns)}",
                data=d.loc[:, channel, :],
                family=sm.families.Gaussian(),
            ).fit()
            _res.append(
                r.summary2().tables[1].assign(variable=var, channel=channel)
            )
    res = pd.concat(_res).drop("Intercept").rename_axis(index="variable")
    res.to_csv(output_dir / "attribute_regression.subcellular_localization.csv")


def correlate_with_RNA_score() -> None:
    """
    Plot fraction of cells per ROI/Sample and grouped by disease/phenotype.
    """
    output_dir = results_dir / "abundance"
    output_dir.mkdir()

    (
        roi_counts,
        roi_areas,
        sample_counts,
        sample_areas,
    ) = generate_count_matrices()

    # Plot score vs CD8
    clinical = pd.read_csv(metadata_dir / "samples.csv", index_col=0)

    sigs = [
        ("ES", "Tcell Inflamation Signature ssGSEA ES"),
        ("ZS", "Tcell Inflamation Signature (z-score)"),
    ]

    for lab, sig in sigs:
        perc = (sample_counts.T / sample_counts.sum(1)).T * 100
        mm2 = (sample_counts.T / sample_areas).T * 1e6
        for df, ttype in [(perc, "percentage"), (mm2, "absolute")]:
            g = df.columns.str.extract(r"\d+ - (.*) \(.*")[0].values
            meta = (
                df.T.groupby(g)
                .agg(np.mean if ttype == "absolute" else np.sum)
                .T
            )
            for df2, label in [(df, "cluster"), (meta, "cell_type")]:
                tp = df2.join(clinical[sig]).dropna()
                fig = get_grid_dims(df2.columns, return_fig=True, sharex=True)
                for i, (ax, col) in enumerate(zip(fig.axes, df2.columns)):
                    statsres = pg.corr(tp[sig], tp[col]).squeeze()
                    ax.scatter(tp[sig], tp[col])
                    sns.regplot(x=tp[sig], y=tp[col], ax=ax)
                    xd = tp[sig].min()
                    xu = tp[sig].max()
                    ax.set(
                        title=f"{col}\nr={statsres['r']:.3f}; p={statsres['p-val']:.3f}",
                        xlim=(xd + (xd * 0.1), xu + (xu * 0.1)),
                        ylabel=None,
                    )
                for ax in fig.axes[i + 1 :]:
                    ax.axis("off")
                fig.savefig(
                    output_dir
                    / f"ssGSEA_score.{lab}.correlation_with_{label}.{ttype}.svg"
                )


def tumor_cell_heterogeneity(a: anndata.AnnData) -> None:
    # Investigate co-expression pattern within tumor cells

    output_dir = results_dir / "tumor_heterogeneity"
    output_dir.mkdir()

    # # I will construct a latent space using only two markers
    # # in cancer cells only
    tumor_markers = ["GATA3(Eu153)", "KRT5(Dy163)"]

    sa = a[
        a.obs["cell_type_broad"].str.contains("Tumor cells"), tumor_markers
    ].copy()
    sa = sa.to_df().groupby(sa.obs["sample"]).mean()
    sas = ((sa - sa.min()) / (sa.max() - sa.min())) + 0.1
    pc_position = np.log(
        sas[tumor_markers[1]] / sas[tumor_markers[0]]
    ).sort_values()
    sa = sa.reindex(pc_position.index)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(
        sa[tumor_markers[1]],
        sa[tumor_markers[0]],
        c=pc_position,
        cmap="coolwarm",
    )
    for i in sa.index:
        ax.text(
            sa.loc[i, tumor_markers[1]],
            sa.loc[i, tumor_markers[0]],
            s=i,
            ha="left",
        )
    ax.set(xlabel="KRT5", ylabel="GATA3")
    fig.savefig(
        output_dir / "Basal_Luminal_axis.only_GATA_KRT5.scatterplot.svg",
        **figkws,
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(pc_position.index, pc_position)
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_xticklabels(pc_position.index, rotation=90)
    ax.set(xlabel="Sample", ylabel="Ba/Sq vs Lum")
    fig.savefig(
        output_dir / "Basal_Luminal_axis.only_GATA_KRT5.rankplot.svg",
        **figkws,
    )

    # # illustrate variability in single cells
    sa = a[
        a.obs["cell_type_broad"].str.contains("Tumor cells"), tumor_markers
    ].copy()
    sa.X = sa.raw[:, tumor_markers].X
    sc.pp.log1p(sa)
    sc.pp.scale(sa)
    sc.pp.pca(sa)
    p = sa.obs[["sample"]].assign(pca=-sa.obsm["X_pca"][:, 0])
    order = p.groupby("sample")["pca"].median().sort_values()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axvline(0, linestyle="--", color="gray")
    sns.violinplot(
        data=p,
        y="sample",
        x="pca",
        orient="horiz",
        order=order.index,
        palette="coolwarm",
        ax=ax,
    )
    ax.set(xlabel="Ba/Sq vs Lum")
    fig.savefig(
        output_dir
        / "Basal_Luminal_axis.only_GATA_KRT5.single_cell.violinplot.svg",
        **figkws,
    )
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.axvline(0, linestyle="--", color="gray")
    sns.boxplot(
        data=p,
        y="sample",
        x="pca",
        orient="horiz",
        order=order.index,
        palette="coolwarm",
        ax=ax,
        fliersize=0,
    )
    ax.set(xlabel="Ba/Sq vs Lum")
    fig.savefig(
        output_dir
        / "Basal_Luminal_axis.only_GATA_KRT5.single_cell.boxplot.svg",
        **figkws,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
