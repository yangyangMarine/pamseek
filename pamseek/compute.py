
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as signal
import matplotlib.dates as mdates
import pytz
from datetime import datetime, timedelta
import shutil

import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# compute PSD rms and percentiles
def compute_rms_psd_and_percentiles(ds):
    """
    Calculate RMS PSD over time for each frequency and percentiles.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Combined Dataset with 'psd_db' (time x frequency) and 'frequency' coordinates.
    
    Returns:
    --------
    f : array-like
        Frequency array from the Dataset.
    rms_psd_db : array-like
        RMS PSD in dB for each frequency.
    percentiles : dict
        Dictionary with percentile keys (1, 5, 50, 95, 99) and corresponding PSD values in dB.
    """
    # Extract frequency and PSD from the Dataset
    f = ds['frequency'].values
    psd_db_array = ds['psd_db'].values
    
    # 1. Compute RMS PSD
    # Convert dB to linear scale (μPa²/Hz)
    psd_linear = 10**(psd_db_array / 10)
    
    # Calculate RMS across time dimension (axis=0)
    rms_psd_linear = np.mean(psd_linear, axis=0)
    
    # Convert back to dB
    rms_psd_db = 10 * np.log10(rms_psd_linear)
    
    # 2. Compute percentiles
    percentile_levels = [1, 5, 50, 95, 99]
    percentiles_db = {
        f"{p}%": 10 * np.log10(np.percentile(psd_linear, p, axis=0))
        for p in percentile_levels
    }

    return f, rms_psd_db, percentiles_db


def segment_bb_spl_metrics(ds, segment_duration='30min'):
    '''
    Calculate median and percentiles of broadband SPL per time segment.
    '''
    # Ensure input is a DataArray of broadband SPL
    if 'bb_spl' not in ds:
        raise ValueError("Input dataset must contain 'bb_spl' variable")
    
    # Extract broadband SPL as DataArray
    bb_spl = ds.bb_spl
    
    # Convert to linear domain for proper statistical calculations
    bb_spl_linear = 10 ** (bb_spl / 10)
    
    # Group by time segments
    grouped_linear = bb_spl_linear.resample(time=segment_duration)
    
    # Calculate percentiles in linear domain, then convert to dB
    percentiles = np.array([
        10 * np.log10(np.percentile(group[1].values, [1, 5, 50, 95, 99]))
        for group in grouped_linear
    ])
    
    # Extract median values separately (they're already in percentiles at index 2)
    median_values = percentiles[:, 2]
    
    # Get segment center times
    segment_times = bb_spl.resample(time=segment_duration).mean().time.values
    
    return (
        segment_times,  # Segment center times
        median_values,  # Median SPL values in dB
        percentiles     # All percentiles [n_segments × 5]
    )
