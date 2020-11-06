"""
Generate simple illustrations of channels and ROIs for visual inspection.
"""

import parmap

from imc.operations import measure_channel_background

from src.init import *


output_dir = results_dir / "illustration"
output_dir.mkdir(exist_ok=True, parents=True)

# QC
# # Mean channels for all ROIs
f = output_dir / prj.name + f".all_rois.mean.pdf"
if not f.exists():
    fig = prj.plot_channels("mean", save=True)
    fig.savefig(f, **figkws)
    plt.close(fig)


# # Signal per channel for all ROIs
for c in tqdm(channel_include):
    f = output_dir / prj.name + f".all_rois.{c}.pdf"
    if f.exists():
        continue
    print(c)
    fig = prj.plot_channels(c, save=True)
    fig.savefig(f, **figkws)
    plt.close(fig)


# # All channels for each ROI
for roi in prj.rois:
    f = output_dir / roi.name + ".all_channels.pdf"
    if f.exists():
        continue
    fig = roi.plot_channels(roi.channel_labels.tolist())
    fig.savefig(f, **figkws)
    plt.close(fig)


# Plot combination of markers
def plot_illustrations(roi, overwrite: bool = True, **kws):
    for colors, chs in illustration_channel_list:
        label = "-".join([f"{k}:{v}" for k, v in zip(colors, chs)])
        _f = output_dir / roi.name + f".{label}.pdf"
        if _f.exists() and not overwrite:
            continue
        _fig = roi.plot_channels(
            chs, output_colors=colors if colors else None, merged=True, **kws
        )
        _fig.savefig(_f, dpi=600, bbox_inches="tight")
        plt.close(_fig)


# Markers
illustration_channel_list = json.load(
    open(metadata_dir / "illustration_markers.json")
)

output_dir = results_dir / "marker_illustration"
output_dir.mkdir(exist_ok=True, parents=True)
parmap.map(plot_illustrations, prj.rois)


# # Segmentation
output_dir = results_dir / "illustration" / "segmentation"
output_dir.mkdir(exist_ok=True, parents=True)
for sample in prj.samples:
    f = output_dir / sample.name + ".probabilities_and_segmentation.pdf"
    if f.exists():
        continue
    fig = sample.plot_probabilities_and_segmentation()
    fig.savefig(f)
    plt.close(fig)


# # Signal
output_dir = results_dir / "marker_illustration"
output_dir.mkdir(exist_ok=True, parents=True)

for sample in prj.samples:
    measure_channel_background(
        sample.rois, output_prefix=output_dir / sample.name
    )
    plt.close("all")
