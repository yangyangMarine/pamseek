"""
Partrac Loggerhead Passive Acoustic Data Processing Script
----------------------------------------------------------
Created: March 26, 2025
Author: Yang Yang
Purpose: This script processes passive acoustic data from Partrac Loggerhead deployments.
         It computes power spectral density (PSD), broadband sound pressure level (SPL),
         and generates visualizations for data analysis.

Inputs:
- Raw acoustic data files stored in the specified `DATA_PATH`

Outputs:
- PSD plot (logarithmic scale)
- Boxplot of broadband SPL over time
- Line plot of broadband SPL trends

"""

# Import necessary libraries
import os
import sys
import glob
import warnings
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Set path for the pamseek repository
PAMSEEK_PATH = r"C:\Users\DrYangYang\Documents\Python\pamseek"
sys.path.append(PAMSEEK_PATH)

# Import custom processing functions from pamseek
from pamseek.batch import (
    compute_audio_file, compute_rms_psd_and_percentiles, plot_psd,
    compute_bb_spl, segment_bb_spl, plot_bb_spl, boxplot_bb_spl
)

# Set data path (Ensure this path exists)
DATA_PATH = r"C:\Users\DrYangYang\Documents\Python\JupyterNotebook\data\example_TCE_LS1X_Site1A_Feb25\2024-09"

# Process audio files to compute PSD and broadband SPL
ds = compute_audio_file(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=None,
    window='hann',
    window_length=1.0,
    overlap=0.5,
    scaling='density',
    low_f=None,
    high_f=None,
    output_dir=DATA_PATH
)

# Compute Power Spectral Density (PSD)
f, rms_psd_db, percentiles = compute_rms_psd_and_percentiles(ds)

# Compute broadband sound pressure level (SPL)
bb = compute_bb_spl(ds)

# Segment broadband SPL over 10-minute intervals
bb_t, bb_rms_spl, bb_percentiles = segment_bb_spl(bb, segment_duration='10min')

# Define output filenames
site_name = "September_Site1A"
psd_plot_filename = f"{site_name}_PSD.png"
spl_boxplot_filename = f"{site_name}_SPL_boxplot.png"
spl_lineplot_filename = f"{site_name}_SPL_line.png"

# Plot PSD
plot_psd(
    f, rms_psd_db, percentiles,
    xscale='log', yscale='linear',
    width=8, height=4, title=site_name,
    grid=True, xlim=None, ylim=None,
    save=True, filename=psd_plot_filename,
    dpi=300, colors=None, xlabel=None, ylabel=None
)

# Create boxplot for broadband SPL
boxplot_bb_spl(
    bb, segment_duration='30min',
    width=8, height=4, title=site_name,
    grid=True, xlim=None, ylim=None,
    save=True, filename=spl_boxplot_filename,
    dpi=300, xlabel=None, ylabel=None,
    showfliers=True, color='grey', box_width=0.6
)

# Plot broadband SPL over time
plot_bb_spl(
    bb_t, bb_rms_spl, bb_percentiles,
    width=8, height=4, title=site_name,
    grid=True, ylim=None,
    save=True, filename=spl_lineplot_filename,
    dpi=300
)

print("Processing complete. Plots saved.")
