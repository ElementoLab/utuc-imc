import json
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from imc import Project
from imc.types import Path

from seaborn_extensions import activate_annotated_clustermap, swarmboxenplot

activate_annotated_clustermap()
swarmboxenplot = partial(swarmboxenplot, test_kws=dict(parametric=False))

MASK_LAYER = "nuclei"

# Set up project
prj = Project(metadata="metadata/samples.csv", name="utuc-imc")
_chs = prj.rois[0].channel_labels.copy()
_chs.index = _chs.values
channel_exclude = _chs[_chs.str.contains("BCK|EMPTY|80Ar|Xe")].tolist()
for _roi in prj.rois:
    _roi.mask_layer = MASK_LAYER
    _roi.set_channel_exclude(channel_exclude)
channel_include = ~_chs.isin(channel_exclude)

excluded_rois = ["20200804_PM784_A1-02"]
for _s in prj:
    _s.rois = [r for r in _s if r.name not in excluded_rois]

# Directories
metadata_dir = Path("metadata")
data_dir = Path("data")
processed_dir = Path("processed")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True, parents=True)


# Output files
metadata_file = metadata_dir / "clinical_annotation.pq"
quantification_file = (
    results_dir / "cell_type" / f"quantification.{MASK_LAYER}.pq"
)
gating_file = results_dir / "cell_type" / f"gating.{MASK_LAYER}.pq"
positive_file = results_dir / "cell_type" / f"gating.positive.{MASK_LAYER}.pq"
positive_count_file = (
    results_dir / "cell_type" / f"gating.positive.count.{MASK_LAYER}.pq"
)
h5ad_file = results_dir / f"utuc-imc.{MASK_LAYER}.h5ad"
counts_file = results_dir / "cell_type" / f"cell_type_counts.{MASK_LAYER}.pq"
counts_agg_file = (
    results_dir / "cell_type" / f"cell_type_counts.{MASK_LAYER}.aggregated_pq"
)
roi_areas_file = results_dir / f"roi_areas.{MASK_LAYER}.csv"
sample_areas_file = results_dir / f"sample_areas.{MASK_LAYER}.csv"


# Plotting parameters
figkws = dict(dpi=300, bbox_inches="tight")


# Metadata
attributes = [
    "Tcell phenotype",
    "Primary/Metastasis",
    "Basal/Luminal",
]
# # ROIs
roi_names = [x.name for x in prj.rois]
roi_attributes = pd.DataFrame(
    np.asarray(
        [getattr(r.sample, attr) for r in prj.rois for attr in attributes]
    ).reshape((-1, len(attributes))),
    index=roi_names,
    columns=attributes,
).rename_axis(index="roi")


# # Samples
sample_names = [x.name for x in prj.samples]
sample_attributes = pd.DataFrame(
    np.asarray(
        [getattr(s, attr) for s in prj.samples for attr in attributes]
    ).reshape((-1, len(attributes))),
    index=sample_names,
    columns=attributes,
).rename_axis(index="sample")

cat_order = {
    "Tcell phenotype": ["Depleted", "Inflamed"],
    "Primary/Metastasis": ["Primary", "Metastasis"],
    "Basal/Luminal": ["Basal", "Luminal"],
}

for _df in [roi_attributes, sample_attributes]:
    for cat, order in cat_order.items():
        _df[cat] = pd.Categorical(_df[cat], categories=order, ordered=True)


# Color codes
colors = dict()
_tab = np.asarray(sns.color_palette("tab10"))
colors["Tcell phenotype"] = _tab[[0, 3]]
colors["Primary/Metastasis"] = _tab[[8, 1]]
colors["Basal/Luminal"] = _tab[[4, 6]]


# Inflamation signature

sig = pd.read_excel("metadata/original/ssgsea_scores_signature_utuc.xlsx")
sig["PM_id"] = (
    "PM"
    + sig["SampleID"].str.extract(r"PM(\d+)")[0].astype(str).str.zfill(4).values
)
sig = sig.query("Cohort.str.contains('EIPM').values", engine="python").drop(
    ["SampleID", "Cohort"], 1
)
sig.groupby("PM_id").mean()
