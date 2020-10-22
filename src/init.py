from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imc import Project
from imc.types import Path


# Set up project
prj = Project(name="utuc-imc")
chs = prj.rois[0].channel_labels
include = pd.Series(~chs.str.contains("BCK|EMPTY|80Ar|Xe").values, index=chs)
exclude = ~include
for roi in prj.rois:
    roi.channels_include = include
    roi.channels_exclude = exclude

channels_include = include[include].index.to_series()
channels_exclude = exclude[exclude].index.to_series()


# Directories
metadata_dir = Path("metadata")
data_dir = Path("data")
processed_dir = Path("processed")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True, parents=True)


# Plotting parameters
figkws = dict(dpi=300, bbox_inches="tight")
