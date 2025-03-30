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

# Enable memory profiling
# %load_ext memory_profiler

# Suppress warnings
warnings.filterwarnings("ignore")

# Set path to read in all the functions 
PAMSEEK_PATH = r"C:\Users\DrYangYang\Documents\Python\pamseek"
sys.path.append(PAMSEEK_PATH)

# Import custom processing functions
from pamseek.batch import (
    process_audio_files, compute_rms_psd_and_percentiles, plot_psd,
    compute_bb_spl, segment_bb_spl, plot_bb_spl, boxplot_bb_spl, chunk_path, compute_toctave_band
)

#####################################
# September data at Site 1A
# Data path, this shoud be processed by month
DATA_PATH = r"D:\TCE_LS1X_Site1A_Feb25\Processed PSD and SPL"

# Process audio files to compute PSD and broadband SPL
ds_full = process_audio_files(
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
    output_dir="September_Full band broadband PSD.nc"
)

# Process audio files to compute PSD and broadband SPL
low_f, high_f = compute_toctave_band(63)
ds_toctave_63Hz = process_audio_files(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=None,
    window='hann',
    window_length=1.0,
    overlap=0.5,
    scaling='density',
    low_f=low_f,
    high_f=high_f,
    output_dir="September_1/3 octave band 63Hz broadband PSD.nc"
)

# Process audio files to compute PSD and broadband SPL
low_f, high_f = compute_toctave_band(125)
ds_toctave_125Hz = process_audio_files(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=None,
    window='hann',
    window_length=1.0,
    overlap=0.5,
    scaling='density',
    low_f=low_f,
    high_f=high_f,
    output_dir="September_1/3 octave band 125Hz broadband PSD.nc"
)

















# Compute Power Spectral Density (PSD)
f, rms_psd_db, percentiles = compute_rms_psd_and_percentiles(ds)

# Compute broadband sound pressure level (SPL)
bb = compute_bb_spl(ds)

# Segment broadband SPL over time intervals, PER DAY
bb_t, bb_rms_spl, bb_percentiles = segment_bb_spl(bb, segment_duration='1D')

###################################################
# Set output dir to save plots 
os.chdir(r'D:\TCE_LS1X_Site1A_Feb25\Output figures\')

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
    showfliers=True, color='grey')

# Plot broadband SPL over time
plot_bb_spl(
    bb_t, bb_rms_spl, bb_percentiles,
    width=8, height=4, title=site_name,
    grid=True, ylim=None,
    save=True, filename=spl_lineplot_filename,
    dpi=300
)

###############################
# get the 1/3 octave band data and redo the process
# 1/3 octave band, 63 Hz
# Process audio files to compute PSD and broadband SPL
low_f, high_f = compute_toctave_band(63)
ds_toctave_63Hz = process_audio_files(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=None,
    window='hann',
    window_length=1.0,
    overlap=0.5,
    scaling='density',
    low_f=low_f,
    high_f=high_f,
    output_dir="Toctave band 63Hz broadband SPL.nc"
)

# Compute broadband sound pressure level (SPL)
bb = compute_bb_spl(ds_toctave_63Hz)
combined_ds.to_netcdf(output_path)
# Segment broadband SPL over time intervals, PER DAY
bb_t, bb_rms_spl, bb_percentiles = segment_bb_spl(bb, segment_duration='12H')

###################################################
# Set output dir to save plots 
os.chdir(r'D:\TCE_LS1X_Site1A_Feb25\Output figures\')

site_name = "September_Site1A"
spl_boxplot_filename = f"{site_name}_63Hz_1/3 octave band_SPL_boxplot.png"
spl_lineplot_filename = f"{site_name}_63Hz_1/3 octave band_SPL_lineplot.png"

# Create boxplot for broadband SPL
boxplot_bb_spl(
    bb, segment_duration='1M',
    width=8, height=4, title=site_name,
    grid=True, xlim=None, ylim=None,
    save=True, filename=spl_boxplot_filename,
    dpi=300, xlabel=None, ylabel="63 Hz 1/3 octave band SPL (dB re 1 uPa)",
    showfliers=False, color='grey')

# Plot broadband SPL over time
plot_bb_spl(
    bb_t, bb_rms_spl, bb_percentiles,
    width=8, height=4, title=site_name,
    grid=True, ylim=None, xlabel=None, ylabel="63 Hz 1/3 octave band SPL (dB re 1 uPa)",
    save=True, filename=spl_lineplot_filename,
    dpi=300)

#####################################
# 125Hz 1/3 band 
###############################
# get the 1/3 octave band data and redo the process
# 1/3 octave band, 125 Hz
# Process audio files to compute PSD and broadband SPL
low_f, high_f = compute_toctave_band(125)
ds_toctave_125Hz = process_audio_files(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=None,
    window='hann',
    window_length=1.0,
    overlap=0.5,
    scaling='density',
    low_f=low_f,
    high_f=high_f,
    output_dir="Toctave band 125Hz broadband SPL.nc"
)

# Compute broadband sound pressure level (SPL)
bb = compute_bb_spl(ds_toctave_125Hz)

# Segment broadband SPL over time intervals, PER DAY
bb_t, bb_rms_spl, bb_percentiles = segment_bb_spl(bb, segment_duration='12H')

###################################################
# Set output dir to save plots 
os.chdir(r'D:\TCE_LS1X_Site1A_Feb25\Output figures\')

site_name = "September_Site1A"
spl_boxplot_filename = f"{site_name}_125Hz_1/3 octave band_SPL_boxplot.png"
spl_lineplot_filename = f"{site_name}_125Hz_1/3 octave band_SPL_lineplot.png"

# Create boxplot for broadband SPL
boxplot_bb_spl(
    bb, segment_duration='1M',
    width=8, height=4, title=site_name,
    grid=True, xlim=None, ylim=None,
    save=True, filename=spl_boxplot_filename,
    dpi=300, xlabel=None, ylabel="125 Hz 1/3 octave band SPL (dB re 1 uPa)",
    showfliers=False, color='grey')

# Plot broadband SPL over time
plot_bb_spl(
    bb_t, bb_rms_spl, bb_percentiles,
    width=8, height=4, title=site_name,
    grid=True, ylim=None, xlabel=None, ylabel="125 Hz 1/3 octave band SPL (dB re 1 uPa)",
    save=True, filename=spl_lineplot_filename,
        dpi=300
    )