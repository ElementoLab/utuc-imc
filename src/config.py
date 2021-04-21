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

from seaborn_extensions import swarmboxenplot

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

include_channels = [
    "AlphaSMA(Pr141)",
    "Vimentin(Nd143)",
    "CD206(Nd144)",
    "CD16(Nd146)",
    "CD163(Sm147)",
    "PanKeratinC11(Nd148)",
    "CD11b(Sm149)",
    "PD1(Nd150)",
    "CD31(Eu151)",
    "CD45(Sm152)",
    "GATA3(Eu153)",
    "FoxP3(Gd155)",
    "CD4(Gd156)",
    "ECadherin(Gd158)",
    "CD68(Tb159)",
    "CD20(Dy161)",
    "CD8a(Dy162)",
    "KRT5(Dy163)",
    "GranzymeB(Er167)",
    "Ki67(Er168)",
    "ColtypeI(Tm169)",
    "CD3(Er170)",
    "CD45RO(Yb173)",
    "PDL1(Lu175)",
    "CD11c(Yb176)",
]
exclude_channels = [
    "80ArAr(ArAr80)",
    "HistoneH3(In113)",
    "<EMPTY>(In115)",
    "129Xe(Xe129)",
    "<EMPTY>(Nd142)",
    "<EMPTY>(Nd145)",
    "<EMPTY>(Sm154)",
    "<EMPTY>(Gd160)",
    "<EMPTY>(Dy164)",
    "<EMPTY>(Ho165)",
    "<EMPTY>(Er166)",
    "<EMPTY>(Yb171)",
    "<EMPTY>(Yb172)",
    "<EMPTY>(Yb174)",
    "190BCKG(BCKG190)",
    "DNA1(Ir191)",
    "DNA2(Ir193)",
    "<EMPTY>(Pb204)",
]
channels_exclude_strings = ["EMPTY", "80ArAr", "129Xe", "190BCKG"]

tech_channels = [
    "DNA",
    "area",
    "perimeter",
    "major_axis_length",
    "eccentricity",
    "solidity",
]
state_channels = [
    "PD1(Nd150)",
    "GranzymeB(Er167)",
    "Ki67(Er168)",
    "PDL1(Lu175)",
]
id_cols = ["sample", "roi", "obj_id"]

t_cell_channels = [
    "CD45(Sm152)",
    "CD45RO(Yb173)",
    "CD3(Er170)",
    "CD4(Gd156)",
    "CD8a(Dy162)",
    "FoxP3(Gd155)",
    "CD11b(Sm149)",
    "GATA3(Eu153)",
    "CD163(Sm147)",
    "PDL1(Lu175)",
    "GranzymeB(Er167)",
    "Ki67(Er168)",
]


# Directories
metadata_dir = Path("metadata")
data_dir = Path("data")
processed_dir = Path("processed")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True, parents=True)


# Output files
metadata_file = metadata_dir / "clinical_annotation.pq"
quantification_file = (
    results_dir / "phenotyping" / f"quantification.{MASK_LAYER}.pq"
)
gating_file = results_dir / "phenotyping" / f"gating.{MASK_LAYER}.pq"
positive_file = results_dir / "phenotyping" / f"gating.positive.{MASK_LAYER}.pq"
positive_count_file = (
    results_dir / "phenotyping" / f"gating.positive.count.{MASK_LAYER}.pq"
)
h5ad_file = results_dir / f"utuc-imc.{MASK_LAYER}.h5ad"
roi_counts_file = (
    results_dir / "phenotyping" / f"cell_type_counts.{MASK_LAYER}.roi.pq"
)
sample_counts_file = (
    results_dir / "phenotyping" / f"cell_type_counts.{MASK_LAYER}.sample.pq"
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
roi_names = pd.MultiIndex.from_tuples(
    [(x.name, x.sample.name) for x in prj.rois], names=["roi", "sample"]
)
roi_attributes = pd.DataFrame(
    np.asarray(
        [getattr(r.sample, attr) for r in prj.rois for attr in attributes]
    ).reshape((-1, len(attributes))),
    # index=roi_names,
    columns=attributes,
).rename_axis(index="roi")
roi_attributes.index = roi_names
roi_attributes = roi_attributes.reset_index(level="sample")


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
