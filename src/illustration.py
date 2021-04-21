#!/usr/bin/env python

"""
Generate simple illustrations of channels and ROIs for visual inspection.
"""

import sys

import parmap
from tqdm import tqdm

from imc.operations import measure_channel_background
from src.config import *

output_dir = results_dir / "illustration"
output_dir.mkdir(exist_ok=True, parents=True)


def main() -> int:
    # QC
    # # Mean channels for all ROIs
    f = output_dir / prj.name + f".all_rois.mean.pdf"
    if not f.exists():
        fig = prj.plot_channels(["mean"], save=True)
        fig.savefig(f, **figkws)
        plt.close(fig)

    # # Signal per channel for all ROIs
    for c in tqdm(channel_include.index[channel_include]):
        f = output_dir / prj.name + f".all_rois.{c}.pdf"
        if f.exists():
            continue
        print(c)
        fig = prj.plot_channels([c], save=True)
        fig.savefig(f, **figkws)
        plt.close(fig)

    # # All channels for each ROI
    for roi in tqdm(prj.rois):
        f = output_dir / roi.name + ".all_channels.pdf"
        if f.exists():
            continue
        fig = roi.plot_channels(roi.channel_labels.tolist())
        fig.savefig(f, **figkws)
        plt.close(fig)

    plot_illustrations()
    example_visualizations(prj)

    return 0


def plot_illustrations() -> None:
    # Markers
    def _plot_illustrations(roi, overwrite: bool = True, **kws):
        for colors, chs in illustration_channel_list:
            label = "-".join([f"{k}:{v}" for k, v in zip(colors, chs)])
            _f = output_dir / roi.name + f".{label}.pdf"
            if _f.exists() and not overwrite:
                continue
            _fig = roi.plot_channels(
                chs,
                target_colors=colors if colors else None,
                merged=True,
                **kws,
            )
            _fig.savefig(_f, dpi=600, bbox_inches="tight")
            plt.close(_fig)

    illustration_channel_list = json.load(
        open(metadata_dir / "illustration_markers.json")
    )

    output_dir = results_dir / "marker_illustration"
    output_dir.mkdir(exist_ok=True, parents=True)
    # _plot_illustrations(prj.rois[0])
    parmap.map(_plot_illustrations, prj.rois)

    roi = prj.rois[25]
    lims = dict(xlim=(367, 1600), ylim=(720, 1500))

    roi = prj.get_rois("20200914_PM1123_A12-02")
    lims = dict(xlim=(100, 600), ylim=(200, 600))

    roi = prj.get_rois("20201001_PM1123_A11-01")
    lims = dict(xlim=(1000, 1800), ylim=(800, 1200))
    lims = dict(xlim=(1200, 2000), ylim=(900, 1400))

    anchor = (lims["xlim"][0], lims["ylim"][0])
    width = lims["xlim"][1] - lims["xlim"][1]
    height = lims["ylim"][1] - lims["ylim"][1]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes[0]):
        roi.plot_channels(
            ["Vimentin", "ECad", "PanKeratin"], merged=True, axes=[ax]
        )
    for i, ax in enumerate(axes[1]):
        roi.plot_channels(["CD3(", "CD16(", "CD206"], merged=True, axes=[ax])
    for ax in axes[:, 1]:
        ax.set(**lims)

    for ax in axes[:, 0]:
        rect = mpl.patches.Rectangle(
            anchor,
            width,
            height,
            linewidth=10,
            edgecolor="white",
            linestyle="--",
        )
        ax.add_patch(rect)
    fig.savefig(results_dir / f"illustration.{roi.name}.pdf", dpi=600)
    fig.savefig(results_dir / f"illustration.{roi.name}.svgz", dpi=600)

    # # Segmentation
    output_dir = results_dir / "illustration" / "segmentation"
    output_dir.mkdir(exist_ok=True, parents=True)
    for sample in prj.samples:
        f = output_dir / sample.name + ".probabilities_and_segmentation.pdf"
        if f.exists():
            continue
        fig = sample.plot_probabilities_and_segmentation()
        fig.savefig(f, **figkws)
        plt.close(fig)

    # # Signal
    output_dir = results_dir / "marker_illustration"
    output_dir.mkdir(exist_ok=True, parents=True)

    for sample in prj.samples:
        measure_channel_background(
            sample.rois, output_prefix=output_dir / sample.name
        )
        plt.close("all")


def example_visualizations(prj) -> None:
    from imc.graphics import get_grid_dims
    from csbdeep.utils import normalize

    output_dir = results_dir / "example_visualizations"
    output_dir.mkdir()

    examples = [
        # ((roi_name, example_name), (pos=((y2, y2), (x2, x1)), markers))
        (
            (
                "20201001_PM1123_A11-01",
                "immune_depleted_global_tumor",
            ),
            (
                None,
                ["Vimentin", "Cadherin", "PanKeratin"],
            ),
        ),
        (
            (
                "20201001_PM1123_A11-01",
                "immune_depleted_global_immune",
            ),
            (
                None,
                ["CD3(", "CD16(", "CD206"],
            ),
        ),
        (
            (
                "20201001_PM1123_A11-01",
                "immune_depleted_global_combined",
            ),
            (
                None,
                ["Cadherin", "CD3(", "DNA"],
            ),
        ),
        (
            (
                "20200914_PM57-02",
                "immune_inflamed_global_tumor",
            ),
            (
                None,
                ["Vimentin", "Cadherin", "PanKeratin"],
            ),
        ),
        (
            (
                "20200914_PM57-02",
                "immune_inflamed_global_immune",
            ),
            (
                None,
                ["CD3(", "CD16(", "CD206"],
            ),
        ),
        (
            (
                "20200914_PM57-02",
                "immune_inflamed_global_combined",
            ),
            (
                None,
                ["Cadherin", "CD3(", "DNA"],
            ),
        ),
        (
            (
                "20200914_PM57-02",
                "tumor_heterog_global1",
            ),
            (
                None,
                ["KRT5", "GATA3", "Vimentin"],
            ),
        ),
        (
            (
                "20200914_PM57-03",
                "tumor_heterog_global2",
            ),
            (
                None,
                ["KRT5", "GATA3", "Vimentin"],
            ),
        ),
        (
            (
                "20200914_PM57-03",
                "tumor_heterog_zoom2",
            ),
            (
                ((850, 350), (1200, 700)),
                ["KRT5", "GATA3", "Vimentin"],
            ),
        ),
        (
            (
                "20200914_PM57-03",
                "tumor_heterog_zoom3",
            ),
            (
                ((1000, 500), (1800, 1300)),
                ["KRT5", "GATA3", "Vimentin"],
            ),
        ),
        (
            (
                "20200923_PM1781_A1",
                "tumor_heterog_lack_global1",
            ),
            (
                None,
                ["KRT5", "GATA3", "Vimentin"],
            ),
        ),
    ]

    for example in examples:
        (roi_name, example_name), (pos, markers) = example
        roi = prj.get_rois(roi_name)
        fig1 = roi.plot_channels(
            markers, equalize=False, position=pos, smooth=1
        )
        fig1.savefig(
            output_dir / f"examples.{example_name}.separate.svg",
            **figkws,
        )

        fig2 = roi.plot_channels(
            markers[:3],
            equalize=False,
            position=pos,
            merged=True,
            # smooth=1
        )
        fig2.savefig(
            output_dir / f"examples.{example_name}.merged.svg", **figkws
        )

        # plot manually
        from imc.graphics import add_scale
        from skimage.filters import gaussian

        p = np.asarray(
            [
                gaussian(normalize(x), sigma=1)
                # normalize(x)
                for x in roi._get_channels(markers[:3])[1]
            ]
        )
        if pos is not None:
            p = p[:, slice(pos[0][1], pos[0][0]), slice(pos[1][1], pos[1][0])]
        fig3, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.moveaxis(normalize(p), 0, -1))
        add_scale(ax)
        ax.axis("off")
        fig3.savefig(
            output_dir / f"examples.{example_name}.merged.smooth.svg",
            **figkws,
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
