#!/usr/bin/env python

"""
Classification of cells into cell types/states.
"""

import re, json

from anndata import AnnData  # type: ignore[import]
import scanpy as sc  # type: ignore[import]
from numpy_groupies import aggregate
from joblib import parallel_backend

from imc.graphics import get_grid_dims, rasterize_scanpy
from imc.utils import align_channels_by_name

from seaborn_extensions import swarmboxenplot

from src.init import *

# from src.operations import (
#     plot_umap,
#     plot_umap_with_labeled_clusters,
#     plot_cluster_heatmaps,
#     plot_cluster_illustrations,
#     plot_cluster_heatmaps_with_labeled_clusters,
# )
# from src.utils import z_score_by_column

sc.settings.n_jobs = -1
cl_resolutions = [1.0, 1.5, 2.0]


# Let's declare some variables for later use
output_dir = results_dir / "cell_type"
output_dir.mkdir()
ids = [
    "roi",
    "sample",
]
min_cells_per_cluster = 500


if not h5ad_file.exists():
    # Quantify single cells
    if not quantification_file.exists():
        prj.quantify_cells()  # by mean in mask layer
        prj.quantification.to_parquet(quantification_file)
    quant = pd.read_parquet(quantification_file).drop(
        channel_exclude, axis=1, errors="ignore"
    )
    # Filter
    cell_filtering_string_query = (
        "area > 5 & area < 300 & "
        "perimeter > 4 & "
        "solidity > 0.6 & "
        "eccentricity > 0.1 & eccentricity < 1.0"
    )
    quant = quant.query(cell_filtering_string_query)

    # To normalize by area
    # c = quant.loc[:, quant.columns.str.contains(r'\(')]
    # c['DNA'] = c.loc[:, c.columns.str.contains(r'DNA')].mean(1)
    # c = c.drop(c.columns[c.columns.str.contains(r'DNA\d\(')], 1)
    # c = (c.T / np.log(quant['area'])).T

    # Let's process these cells and get their cell type identities through clustering
    # # This is roughly equivalent to what is in `src.operations.process_single_cell`,
    # # but we'll use some of the intermediate steps later too.
    obs = (
        quant[["roi", "sample"]]
        .merge(roi_attributes, left_on="roi", right_index=True)
        .rename_axis(index="cell")
        .reset_index(drop=True)
    ).assign(obj_id=quant.index.tolist())

    ann = AnnData(quant.drop(ids, 1).reset_index(drop=True), obs=obs)
    ann.raw = ann
    sc.pp.log1p(ann)

    sc.pp.pca(ann)
    with parallel_backend("threading", n_jobs=12):
        sc.pp.neighbors(ann, n_neighbors=15)
    for res in cl_resolutions:
        sc.tl.leiden(ann, key_added=f"cluster_{res}", resolution=res)
    # 1-based cluster numbering
    for cres in ann.obs.columns.to_series().filter(like="cluster"):
        if ann.obs[cres].astype(int).min() == 0:
            ann.obs[cres] = (ann.obs[cres].astype(int) + 1).astype(str).values
    sc.tl.umap(ann)
    sc.tl.diffmap(ann)
    ann.write_h5ad(h5ad_file)

ann = sc.read_h5ad(h5ad_file)


# Add cluster/metacluster labels
cluster_labels = json.load(open(metadata_dir / "cluster_labels.json"))
cl_labels = [
    str(r) + "_labels" for r in cl_resolutions if str(r) in cluster_labels
]
for res in cl_resolutions:
    if str(res) in cluster_labels:
        ann.obs[f"cluster_{res}_labels"] = ann.obs[f"cluster_{res}"].replace(
            cluster_labels[str(res)]
        )
        ann.obs[f"metacluster_{res}_labels"] = ann.obs[
            f"cluster_{res}_labels"
        ].str.extract(r"\d+ - (.*) \(")[0]


# Add predefined color sets for each attribute
for attr in attributes:
    ann.uns[attr + "_colors"] = np.asarray(
        list(map(mpl.colors.rgb2hex, colors.get(attr)))
    )


# Plot latent representations
# # but first let's increase contrast by capping the intensity values
a = ann.copy()
for ch in a.var.index:
    x = pd.Series(a[:, ch].X.squeeze())
    v = x.quantile(0.99)
    a[:, ch].X = x.clip(upper=v).values
a = a[a.obs.index.to_series().sample(frac=1).values, :]


for algo in ["umap", "pca", "diffmap"]:
    f = getattr(sc.pl, algo)

    # # all markers
    axes = f(
        a,
        color=attributes + ["sample"] + a.var.index.tolist(),
        use_raw=False,
        show=False,
    )
    fig = axes[0].figure
    rasterize_scanpy(fig)
    fig.savefig(output_dir / f"{algo}.all_markers.clipped.svgz", **figkws)

    for res in cl_resolutions + cl_labels:  # type: ignore
        # # just clusters with centroids
        ax = f(a, color=f"cluster_{res}", show=False, s=0.5, alpha=0.9)
        fig = ax.figure
        rasterize_scanpy(fig)
        # # # centroids:
        cent = aggregate(
            a.obs[f"cluster_{res}"].cat.codes,
            a.obsm[f"X_{algo}"][:, :2],
            func="mean",
            axis=0,
        )
        for i, clust in enumerate(
            a.obs[f"cluster_{res}"].sort_values().unique()
        ):
            ax.text(*cent[i], s=clust)
        fig.savefig(output_dir / f"{algo}_{res}.clipped.pdf", **figkws)

        # # just attributes
        ax = f(a, color=attributes, show=False, s=0.5, alpha=0.9)
        fig = ax[0].figure
        rasterize_scanpy(fig)
        fig.savefig(
            output_dir / f"{algo}_{res}.attributes.clipped.svgz", **figkws
        )

        # # clusters separately
        aa = a.copy()
        d = pd.get_dummies(a.obs[f"cluster_{res}"])
        d.columns = "C" + d.columns.astype(str).str.zfill(2)
        aa.obs = aa.obs.join(d)
        axes = f(aa, color=d.columns.tolist(), cmap="binary", show=False,)
        fig = axes[0].figure
        rasterize_scanpy(fig)
        fig.savefig(
            output_dir / f"{algo}_{res}.all_markers.clipped.clusters.svgz",
            **figkws,
        )


# Plot mean phenotypes for each cluster
for res in cl_resolutions + cl_labels:  # type: ignore
    df = a.to_df().groupby(a.obs[f"cluster_{res}"]).mean()

    # keep clusters with only 500 cells
    total = a.obs[f"cluster_{res}"].value_counts().rename("cells")
    total = total.loc[lambda x: x > min_cells_per_cluster]
    att_count = (
        a.obs.groupby(attributes)[f"cluster_{res}"]
        .value_counts()
        .rename("cells")
    )
    att_count = (
        att_count.reset_index()
        .pivot_table(
            index=f"cluster_{res}",
            columns=attributes,
            values="cells",
            aggfunc=sum,
            fill_value=0,
        )
        .loc[total.index]
    )
    df = df.loc[total.index]

    kws = dict(
        robust=True,
        yticklabels=True,
        cbar_kws=dict(label="Z-score"),
        row_colors=att_count.join(total),
        figsize=(15, 10),
        dendrogram_ratio=0.075,
        metric="correlation",
    )
    grid = sns.clustermap(df, **kws)
    grid.savefig(output_dir / f"cluster_mean_{res}.clustermap.svg")

    for pat, label in [("", "all"), (r"\(", "no_struct")]:
        p = df.loc[:, df.columns.str.contains(pat)]
        grid = sns.clustermap(p, z_score=1, center=0, cmap="RdBu_r", **kws)
        grid.savefig(
            output_dir / f"cluster_mean_{res}.clustermap.{label}.z_score.svg"
        )

        grid2 = sns.clustermap(
            p, z_score=1, center=0, cmap="RdBu_r", row_cluster=False, **kws
        )
        grid2.savefig(
            output_dir
            / f"cluster_mean_{res}.clustermap.{label}.z_score.cluster_sorted.svg"
        )

    fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    sns.heatmap(
        att_count.iloc[grid.dendrogram_row.reordered_ind] + 1,
        cmap="PuOr_r",
        ax=ax,
        norm=mpl.colors.LogNorm(),
        cbar_kws=dict(label="Cells"),
    )
    fig.savefig(
        output_dir / f"cluster_mean_{res}.clustermap.attribute_abundance.svg"
    )


# Choose one clustering resolution
res = 1.5

# Let's generate some dataframes that will be useful later:
# # counts of cells per image, per cluster
roi_counts = (
    ann.obs[["roi", f"cluster_{res}_labels"]]
    .assign(count=1)
    .pivot_table(
        index="roi",
        columns=f"cluster_{res}_labels",
        values="count",
        aggfunc=sum,
        fill_value=0,
    )
)
roi_counts = roi_counts.loc[:, ~roi_counts.columns.str.contains(r"\?")]
roi_counts.to_parquet(counts_file)

# # counts of cells per image, per cluster, for meta-clusters
agg_counts = (
    roi_counts.T.groupby(
        roi_counts.columns.str.extract(r"\d+ - (.*) \(")[0].values
    )
    .sum()
    .T
)
agg_counts.to_parquet(counts_agg_file)

# # Area per image
roi_areas = pd.Series(
    {r.name: r.area for r in prj.rois}, name="area"
).rename_axis("roi")
roi_areas.to_csv(roi_areas_file)

# # counts of cells per sample, per cluster
sample_counts = (
    ann.obs[["sample", f"cluster_{res}_labels"]]
    .assign(count=1)
    .pivot_table(
        index="sample",
        columns=f"cluster_{res}_labels",
        values="count",
        aggfunc=sum,
        fill_value=0,
    )
)
sample_counts = sample_counts.loc[:, ~sample_counts.columns.str.contains(r"\?")]

# # Area per sample
sample_areas = pd.Series(
    {s.name: sum([r.area for r in s]) for s in prj}, name="area"
)
sample_areas.to_csv(sample_areas_file)


# Plot fraction of cells per ROI/Sample and grouped by disease/phenotype
prefix = "nuclei."

# # Heatmaps
for grouping, df, area, attrs in [
    ("sample", sample_counts, sample_areas, sample_attributes),
    ("roi", roi_counts, roi_areas, roi_attributes),
]:
    attrs = sample_attributes if grouping == "sample" else roi_attributes
    kws = dict(
        figsize=(16, 8),
        rasterized=True,
        metric="correlation",
        col_colors=attrs,
        yticklabels=True,
    )
    grid = sns.clustermap((df.T / df.sum(1)), **kws)
    grid.savefig(
        output_dir / f"clustering.{prefix}cells_per_{grouping}.{res}.svg",
        **figkws,
    )
    grid = sns.clustermap((df.T / df.sum(1)), standard_scale=0, **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.standard_scale.svg",
        **figkws,
    )

    df2 = df.copy()
    df2.columns = df2.columns.str.extract(r"\d+ - (.*) \(")[0]
    df2 = df2.T.groupby(level=0).sum().T.rename_axis(columns="Cluster")
    kws = dict(
        figsize=(10, 8),
        rasterized=True,
        metric="correlation",
        col_colors=attrs,
        yticklabels=True,
    )
    grid = sns.clustermap((df2.T / df2.sum(1)), **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.reduced_clusters.svg",
        **figkws,
    )
    grid = sns.clustermap((df2.T / df2.sum(1)), standard_scale=0, **kws)
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.reduced_clusters.standard_scale.svg",
        **figkws,
    )

    dfarea = (df.T / area).T * 1e6
    grid = sns.clustermap(
        dfarea, xticklabels=True, row_colors=attrs, figsize=(10, 12)
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.per_area.svg",
        **figkws,
    )

    dfarea_red = (
        dfarea.T.groupby(dfarea.columns.str.extract(r"\d+ - (.*)\(")[0].values)
        .sum()
        .T
    )
    grid = sns.clustermap(
        dfarea_red,
        metric="correlation",
        xticklabels=True,
        row_colors=attrs,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.per_area.reduced_clusters.svg",
        **figkws,
    )
    grid = sns.clustermap(
        np.log1p(dfarea_red),
        metric="correlation",
        xticklabels=True,
        row_colors=attrs,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.per_area.reduced_clusters.log1p.svg",
        **figkws,
    )
    grid = sns.clustermap(
        np.log1p(dfarea_red),
        z_score=1,
        center=0,
        cmap="RdBu_r",
        robust=True,
        metric="correlation",
        xticklabels=True,
        row_colors=attrs,
        figsize=(10, 6),
    )
    grid.savefig(
        output_dir
        / f"clustering.{prefix}cells_per_{grouping}.{res}.per_area.reduced_clusters.log1p.zscore.svg",
        **figkws,
    )

# # Swarmboxenplots
_stats = list()
for grouping, df, area in [
    ("sample", sample_counts, sample_areas),
    ("roi", roi_counts, roi_areas),
]:
    attrs = sample_attributes if grouping == "sample" else roi_attributes

    for measure, df2 in [
        ("percentage", (df.T / df.sum(1)).T * 100),
        ("area", (df.T / area).T * 1e6),
    ]:
        df2.index.name = grouping
        df2 = (
            df2.join(attrs[attributes])
            .reset_index()
            .melt(
                id_vars=[grouping] + attributes,
                value_name=measure,
                var_name="cell_type",
            )
        )
        for var in attributes:
            cts = df2["cell_type"].unique()
            n = len(cts)
            n, m = get_grid_dims(n)
            fig, axes = plt.subplots(
                n, m, figsize=(m * 3, n * 3), sharex=True, tight_layout=True
            )
            axes = axes.flatten()
            for i, ct in enumerate(cts):
                sts = swarmboxenplot(
                    data=df2.query(f"cell_type == '{ct}'"),
                    x=var,
                    y=measure,
                    test_kws=dict(parametric=False),
                    plot_kws=dict(palette=colors[var]),
                    ax=axes[i],
                )
                _stats.append(
                    sts.assign(
                        grouping=grouping,
                        variable="original",
                        cell_type=ct,
                        measure=measure,
                    )
                )
                axes[i].set(title="\n(".join(ct.split(" (")))
            for ax in axes[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_dir
                / f"clustering.{prefix}fraction.{cluster_str}.cells_per_{grouping}.by_{var.replace('/', '-')}.{measure}.svg",
                **figkws,
            )
            plt.close(fig)

            df3 = df2.copy()
            df3["cell_type"] = df3["cell_type"].str.extract(r"\d+ - (.*) \(")[0]
            df3 = (
                df3.groupby(attributes + [grouping, "cell_type"])
                .sum()
                .reset_index()
                .dropna()
            )
            cts = df3["cell_type"].unique()
            n = len(cts)
            n, m = get_grid_dims(n)
            fig, axes = plt.subplots(
                n, m, figsize=(m * 3, n * 3), sharex=True, tight_layout=True
            )
            axes = axes.flatten()
            for i, ct in enumerate(cts):
                sts = swarmboxenplot(
                    data=df3.query(f"cell_type == '{ct}'"),
                    x=var,
                    y=measure,
                    test_kws=dict(parametric=False),
                    plot_kws=dict(palette=colors[var]),
                    ax=axes[i],
                )
                _stats.append(
                    sts.assign(
                        grouping=grouping,
                        variable="aggregated",
                        cell_type=ct,
                        measure=measure,
                    )
                )
                axes[i].set(title=ct)
            for ax in axes[i + 1 :]:
                ax.axis("off")
            fig.savefig(
                output_dir
                / f"clustering.{prefix}fraction.{cluster_str}.cells_per_{grouping}.reduced_clusters.by_{var.replace('/', '-')}.{measure}.svg",
                **figkws,
            )
            plt.close(fig)

stats = pd.concat(_stats)
stats.to_csv(
    output_dir / f"clustering.{prefix}fraction.{cluster_str}.differences.csv",
    index=False,
)
# # Save for supplement
# df2 = stats.loc[~stats["cell_type"].str.contains(r"\?")]
# df2.to_excel("manuscript/Supplementary Table 3.xlsx", index=False)


# Illustrate clusters

# plot_cluster_illustrations(ann)


# plot percentages
c = (agg_counts.T / roi_areas).T * 1e6

fig, axes = plt.subplots(2, 5)
for x, ax in zip(c.columns, axes.flat):
    swarmboxenplot(
        data=c[[x]].join(roi_attributes),
        x=attributes[0],
        y=x,
        ax=ax,
        plot_kws=dict(palette=colors[attributes[0]]),
    )
fig.savefig(results_dir / "abundance.absolute.svg", **figkws)

perc = ((agg_counts.T / agg_counts.sum(1)) * 100).T

fig, axes = plt.subplots(2, 5)
for x, ax in zip(perc.columns, axes.flat):
    swarmboxenplot(
        data=perc[[x]].join(roi_attributes),
        x=attributes[0],
        y=x,
        ax=ax,
        plot_kws=dict(palette=colors[attributes[0]]),
    )
fig.savefig(results_dir / "abundance.percentage.svg", **figkws)


p1 = perc.groupby(perc.index.str.extract("(.*)-")[0].values).mean()

structural = ["Endothelial cells", "Fibroblasts", "Smooth muscle cells"]
p1["Structural cells"] = p1[structural].sum(1)
p1["Immune cells"] = 100 - p1[["Structural cells", "Tumor cells"]].sum(1)
p1 = p1.sort_values(["Immune cells", "Tumor cells"])[
    ["Tumor cells", "Structural cells", "Immune cells"]
]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
p1.plot(kind="bar", stacked=True, ax=axes[0])
p2 = perc.groupby(perc.index.str.extract("(.*)-")[0].values).mean()
p2 = p2.drop(["Tumor cells"] + structural, 1)
p2 = (p2.T / p2.sum(1)).T.reindex(p1.index) * 100
p2.plot(kind="bar", stacked=True, ax=axes[1])
fig.savefig(
    results_dir / "cell_type_composition.percentage.stacked_barplot.svg"
)


# Plot score vs CD8
sig = "Tcell Inflamation Signature ssGSEA ES"

clinical = pd.read_csv(metadata_dir / "samples.csv", index_col=0)

c1 = c.groupby(c.index.str.extract("(.*)-")[0].values).mean()
tp = c1.join(clinical[sig]).dropna()

fig, axes = plt.subplots(3, 3, figsize=(7, 7))
axes = axes.flatten()
for i, col in enumerate(c1.columns):
    axes[i].scatter(tp[sig], tp[col])
    axes[i].set_title(col)
fig.savefig(
    results_dir / "ssGSEA_score.correlation_with_cell_types.absolute.svg"
)

p1 = perc.groupby(perc.index.str.extract("(.*)-")[0].values).mean()
tp = p1.join(clinical[sig]).dropna()
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
axes = axes.flatten()
for i, col in enumerate(p1.columns):
    axes[i].scatter(tp[sig], tp[col])
    axes[i].set_title(col)
fig.savefig(
    results_dir / "ssGSEA_score.correlation_with_cell_types.percentage.svg"
)


#

#

#

#

#

#

#

# # Recluster non-tumor cells
# ann = sc.read_h5ad(h5ad_file)

# # Add cluster/metacluster labels
# cluster_labels = json.load(open(metadata_dir / "cluster_labels.json"))
# resolution = 1.0
# ann.obs[f"cluster_{resolution}_labels"] = ann.obs[
#     f"cluster_{resolution}"
# ].replace(cluster_labels[str(resolution)])
# ann.obs[f"metacluster_{resolution}_labels"] = ann.obs[
#     f"cluster_{resolution}_labels"
# ].str.extract(r"\d+ - (.*) \(")[0]

# exclude = [
#     "Tumor cells",
#     "No marker",
#     "Small cells",
#     "Fibroblasts",
#     "Smooth muscle cells",
#     "Endothelial cells",
# ]
# ann = ann[~ann.obs[f"metacluster_{resolution}_labels"].isin(exclude), :]


# for ch in ann.var.index:
#     x = pd.Series(ann[:, ch].X.squeeze())
#     v = x.quantile(0.99)
#     ann[:, ch].X = x.clip(upper=v).values
# ann = ann[ann.obs.index.to_series().sample(frac=1).values, :]

# sc.pp.pca(ann)
# sc.pp.neighbors(ann, n_neighbors=15)
# sc.tl.umap(ann)
# for res in [1.0, 2.0]:
#     sc.tl.leiden(ann, resolution=res, key_added=f"immune_cluster_{res}")


# # Plot mean phenotypes for each cluster
# a = ann
# for res in ["1.0", "2.0"]:
#     df = a.to_df().groupby(a.obs[f"immune_cluster_{res}"]).mean()

#     # keep clusters with only 500 cells
#     total = a.obs[f"immune_cluster_{res}"].value_counts().rename("cells")
#     total = total.loc[lambda x: x > min_cells_per_cluster]
#     att_count = (
#         a.obs.groupby(attributes)[f"immune_cluster_{res}"]
#         .value_counts()
#         .rename("cells")
#     )
#     att_count = (
#         att_count.reset_index()
#         .pivot_table(
#             index=f"immune_cluster_{res}",
#             columns=attributes,
#             values="cells",
#             aggfunc=sum,
#             fill_value=0,
#         )
#         .loc[total.index]
#     )
#     df = df.loc[total.index]

#     kws = dict(
#         robust=True,
#         yticklabels=True,
#         cbar_kws=dict(label="Z-score"),
#         row_colors=att_count.join(total),
#         figsize=(15, 10),
#         dendrogram_ratio=0.075,
#         metric="correlation",
#     )
#     grid = sns.clustermap(df, **kws)
#     grid.savefig(output_dir / f"immune_cluster_mean_{res}.clustermap.svg")

#     grid = sns.clustermap(df, z_score=1, center=0, cmap="RdBu_r", **kws)
#     grid.savefig(
#         output_dir / f"immune_cluster_mean_{res}.clustermap.z_score.svg"
#     )

#     grid2 = sns.clustermap(
#         df, z_score=1, center=0, cmap="RdBu_r", row_cluster=False, **kws
#     )
#     grid2.savefig(
#         output_dir
#         / f"immune_cluster_mean_{res}.clustermap.z_score.immune_cluster_sorted.svg"
#     )

#     fig, ax = plt.subplots(1, 1, figsize=(4, 10))
#     sns.heatmap(
#         att_count.iloc[grid.dendrogram_row.reordered_ind] + 1,
#         cmap="PuOr_r",
#         ax=ax,
#         norm=mpl.colors.LogNorm(),
#         cbar_kws=dict(label="Cells"),
#     )
#     fig.savefig(
#         output_dir
#         / f"immune_cluster_mean_{res}.clustermap.attribute_abundance.svg"
#     )


# immune_markers = ann.var.index[ann.var.index.str.startswith("CD")].tolist()
# immune_markers += ["cluster_1.0_labels"]
# immune_markers += [f"immune_cluster_{res}" for res in [1.0, 2.0]]


# # sc.pl.pca(ann, color=immune_markers, use_raw=False)
# # sc.pl.umap(ann, color=immune_markers, use_raw=False)

# # # all markers
# ann.obs = (
#     ann.obs.reset_index()
#     .set_index("sample")
#     .join(score)
#     .rename_axis("sample")
#     .reset_index()
#     .set_index("index")
#     .reindex(ann.obs.index)
# )

# cluster_strs = ann.obs.columns[ann.obs.columns.str.contains("cluster")].tolist()

# algo = "umap"
# axes = sc.pl.umap(
#     ann,
#     color=attributes
#     + ["sample", score.name]
#     + cluster_strs
#     + a.var.index.tolist(),
#     use_raw=False,
#     show=False,
# )
# fig = axes[0].figure
# rasterize_scanpy(fig)
# fig.savefig(
#     output_dir / f"{algo}_all_res.immune.all_markers.clipped.svgz", **figkws
# )


# org = sc.read_h5ad(h5ad_file)
# cluster_strs = ann.obs.columns[ann.obs.columns.str.contains("cluster")].tolist()

# score = pd.read_csv("metadata/samples.csv", index_col=0)[
#     "Tcell Inflamation Signature (z-score)"
# ]

# _res = list()
# for clust in cluster_strs:
#     counts = ann.obs.groupby("sample")[clust].value_counts()

#     perc = (counts / counts.groupby(level=0).sum()) * 100
#     p = perc.rename("percentage").reset_index(level=1).join(score)
#     for c in p[clust].unique():
#         _res.append(
#             [clust, c, p.loc[p[clust] == c].drop(clust, 1).corr().iloc[1, 0],]
#         )
# res = (
#     pd.DataFrame(_res, columns=["clustering", "cluster", "corr"])
#     .dropna()
#     .sort_values("corr")
# )
# res.to_csv("res.csv", index=False)


# res.loc[res["cluster"].str.contains(" - ")].drop("clustering", 1)

# clust = "cluster_1.0_labels"
# counts = ann.obs.groupby("sample")[clust].value_counts()
# perc = (counts / counts.groupby(level=0).sum()) * 100

# p = perc.rename("percentage").reset_index(level=1).join(score)

# fig, axes = plt.subplots(2, 5, figsize=(5 * 3, 2 * 3))
# axes = axes.flatten()
# for i, c in enumerate(p[clust].unique()):
#     pp = p.loc[p[clust] == c]
#     axes[i].scatter(np.log1p(pp["percentage"]), pp[score.name])
#     axes[i].set(title=c)
