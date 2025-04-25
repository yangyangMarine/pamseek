import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import scipy.signal as signal
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta
import shutil
import netCDF4

import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from opensoundscape import Audio

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename in format '20240717T093000_...' for loggerhead hydrophone.
    
    Parameters:
    -----------
    filename : str
        Filename to extract timestamp from
        
    Returns:
    --------
    datetime
        Datetime object representing the timestamp
    """
    try:
        # Extract the timestamp part before the first underscore
        timestamp_part = filename.split('_')[0]
        
        # Parse the timestamp string into a datetime object
        # Format: YYYYMMDDTHHMMSS
        date_part = timestamp_part[:8]  # YYYYMMDD
        time_part = timestamp_part[9:]  # HHMMSS (skipping the 'T')
        
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6]) if len(time_part) >= 6 else 0
        
        return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse timestamp from filename {filename}: {e}")
        # Return a default timestamp if parsing fails
        return datetime(1970, 1, 1, 0, 0, 0)

def calibrate_hydrophone_signal(audio_data, sensitivity_db, gain=0, bit_depth=16):
    """
    Correct a raw hydrophone audio signal by applying sensitivity, gain, and bit depth correction.
    
    Parameters:
    -----------
    audio_data : opensoundscape.Audio or numpy.ndarray
        Audio object or 1-dimensional array of raw digital wave signal
    sensitivity_db : float
        Hydrophone sensitivity in dB re 1V/µPa
    gain : float, optional
        Additional gain applied to the signal in dB (default: 0 dB)
    bit_depth : int, optional
        Bit depth of the audio signal (default: 16)
        
    Returns:
    --------
    opensoundscape.Audio or numpy.ndarray
        Audio object or array with corrected signal in Pascals (Pa)
    """
    # Handle either Audio object or numpy array
    is_audio_object = hasattr(audio_data, 'samples')
    samples = audio_data.samples if is_audio_object else audio_data
    
    # Normalize raw data based on bit depth
    if bit_depth == 16:
        normalized_signal = samples / 32768.0
    elif bit_depth == 24:
        normalized_signal = samples / 8388608.0
    elif bit_depth == 32:
        if np.issubdtype(samples.dtype, np.integer):
            normalized_signal = samples / 2147483648.0
        else:
            normalized_signal = samples
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    # Convert sensitivity from dB to linear scale
    sensitivity_linear = 10**(sensitivity_db/20)
    
    # Convert sensitivity to Pa/V (reciprocal of V/Pa)
    receive_sensitivity = 1 / sensitivity_linear
    
    # Apply sensitivity and gain to convert voltage to pressure (Pa)
    gain_linear = 10**(gain/20)
    sound_pressure = normalized_signal * receive_sensitivity * gain_linear
    
    # Return the appropriate type
    if is_audio_object:
        audio_data.samples = sound_pressure
        return audio_data
    else:
        return sound_pressure

def list_files(datapath, keyword):
    """
    Lists all files in the specified directory that contain the given keyword in their filename.
    
    Args:
        datapath (str): Path to the directory to search in.
        keyword (str): Keyword to filter filenames by.
    
    Returns:
        list: A list of full paths to files containing the keyword in their names.
        
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        TypeError: If the inputs are not strings.
    """
    # Check input types
    if not isinstance(datapath, str):
        raise TypeError("datapath must be a string")
    if not isinstance(keyword, str):
        raise TypeError("keyword must be a string")
    
    # Check if directory exists
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Directory not found: {datapath}")
    if not os.path.isdir(datapath):
        raise NotADirectoryError(f"Path is not a directory: {datapath}")
    
    return [os.path.join(datapath, f) for f in os.listdir(datapath) if keyword in f]


def process_audio_files(path, sensitivity, gain, fs=None, 
                         window='hann', window_length=1.0, overlap=0.5, 
                         scaling='density', low_f=None, high_f=None, 
                         output_dir=None, output_filename=None):
    """
    Process multiple WAV files in a directory to compute Power Spectral Density (PSD).
    """
    # Suppress numpy runtime warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Store original working directory
    original_dir = os.getcwd()
    try:
        # Change to input path
        os.chdir(path)
        
        # Set output directory
        output_dir = output_dir or path
        os.makedirs(output_dir, exist_ok=True)

        # Find WAV files
        wav_files = sorted(glob.glob('*.wav'))
        if not wav_files:
            raise ValueError(f"No .wav files found in {path}")

        # Reference pressure and epsilon for calculations
        P_REF = 1e-6  # reference pressure
        EPSILON = np.finfo(float).eps

        # Containers for combined results
        combined_f = []
        combined_t = []
        combined_psd_db = []
        combined_bb_spl_db = []

        # Track processed and skipped files
        processed_files = []
        skipped_files = []

        # Process each WAV file
        for i, single_file in enumerate(wav_files, 1):
            # Progress tracking
            progress = int((i/len(wav_files))*100)
            file_count = f"({i}/{len(wav_files)})"
            progress_bar = "[" + "=" * (progress//2) + " " * (50 - progress//2) + "]"
            print(f"\rProcessing {file_count} {progress_bar} {progress}%", end="", flush=True)
            
            try:
                # Load audio file
                audio_object = Audio.from_file(single_file)

                # Extract timestamp
                timestamp_str = extract_timestamp_from_filename(single_file)
                start_time = pytz.timezone("UTC").localize(timestamp_str)

                # Calibrate hydrophone signal
                audio_object = calibrate_hydrophone_signal(audio_object, sensitivity, gain, bit_depth=16)

                # Apply optional bandpass filter
                samples = (audio_object.bandpass(low_f=low_f, high_f=high_f, order=4).samples 
                            if low_f is not None and high_f is not None 
                            else audio_object.samples)

                # Determine sample rate
                sample_rate = audio_object.sample_rate if fs is None else fs
                
                # Compute Welch's PSD
                nperseg = int(sample_rate * window_length)
                noverlap = int(nperseg * overlap)

                f, psd_welch = signal.welch(
                    samples, 
                    fs=sample_rate,
                    window=window, 
                    nperseg=nperseg,
                    noverlap=noverlap, 
                    scaling=scaling
                )
                
                # Convert PSD to dB
                psd_db = 10 * np.log10(psd_welch / (P_REF**2))
                
                # Compute broadband SPL
                power_total = np.trapz(psd_welch, f)
                bb_spl_db = 10 * np.log10(power_total / (P_REF**2) + EPSILON)
                
                # Collect data for combined dataset
                combined_f = f  # Frequency array is the same for all files
                start_time_utc = start_time.astimezone(pytz.UTC).replace(tzinfo=None)
                combined_t.append(np.datetime64(start_time_utc))

                combined_psd_db.append(psd_db)
                combined_bb_spl_db.append(bb_spl_db)
                
                processed_files.append(single_file)

            except Exception as e:
                print(f"\nError processing {single_file}: {e}")
                skipped_files.append((single_file, str(e)))
                continue

        # Print processing summary
        print("\n\nProcessing Summary:")
        print(f"Total files: {len(wav_files)}")
        print(f"Processed files: {len(processed_files)}")
        print(f"Skipped files: {len(skipped_files)}")
        
        if skipped_files:
            print("\nSkipped Files:")
            for file, error in skipped_files:
                print(f"- {file}: {error}")

        # Check if any files were processed
        if not processed_files:
            print("No files could be processed. Returning None.")
            return None

        # Create combined xarray Dataset
        output_path = os.path.join(output_dir, output_filename)
        combined_ds = xr.Dataset(
            {
                'psd_db': (['time', 'frequency'], np.array(combined_psd_db)),
                'bb_spl_db': (['time'], combined_bb_spl_db),
            },
            coords={
                'frequency': combined_f,
                'time': np.array(combined_t, dtype='datetime64[ns]')
            },
            attrs={
                'sensitivity': sensitivity,
                'gain': gain,
                'sample_rate': sample_rate,
                'window': window,
                'window_length': window_length,
                'overlap': overlap,
                'scaling': scaling,
                'low_f': low_f if low_f is not None else 'None',
                'high_f': high_f if high_f is not None else 'None',
                'processed_files': processed_files,
                'skipped_files': [f[0] for f in skipped_files]
            }
        )
        
        # Save combined dataset
        # Then save with explicit encoding for any Unicode variables, to deal with netCDF4 issues, ('U', 224 error)
        encoding = {var: {'dtype': 'str'} for var in combined_ds.variables if combined_ds[var].dtype.kind == 'U'}
        combined_ds.to_netcdf(output_path, engine='netcdf4',encoding=encoding)
        print(f"\nCombined dataset saved to {output_path}")

    except Exception as e:
            print(f"Unexpected error in processing: {e}")
            return None

    return combined_ds
    os.chdir(original_dir)

def process_audio_files_chunked(file_dict, sensitivity, gain, base_output_dir=None, fs=None, 
                         window='hann', window_length=1.0, overlap=0.5, 
                         scaling='density', low_f=None, high_f=None):
    """
    Process multiple WAV files from a provided dictionary of file lists.
    
    Parameters:
    -----------
    file_dict : dict
        Dictionary where keys are output filenames and values are lists of full file paths
    sensitivity : float
        Hydrophone sensitivity
    gain : float
        Hydrophone gain
    base_output_dir : str, optional
        Base directory where output files will be saved. 
        If None, uses the directory of the first file.
    ... (other parameters remain the same as in original function)
    
    Returns:
    --------
    dict
        Dictionary of xarray Datasets, with keys corresponding to input dictionary keys
    """
    # Suppress numpy runtime warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Store original working directory
    original_dir = os.getcwd()
    
    # Validate and prepare output
    output_datasets = {}
    
    try:
        # Validate input dictionary
        if not file_dict or not all(isinstance(files, list) for files in file_dict.values()):
            raise ValueError("Input must be a dictionary with lists of .wav file paths")

        # Determine base output directory
        if base_output_dir is None:
            # Use directory of first file from first list in the dictionary
            first_list = next(iter(file_dict.values()))
            base_output_dir = os.path.dirname(first_list[0])

        # Process each list of files
        for output_filename, file_list in file_dict.items():
            # Ensure filename ends with .nc
            if not output_filename.endswith('.nc'):
                output_filename += '.nc'

            # Validate file list
            if not file_list or not all(file.lower().endswith('.wav') for file in file_list):
                print(f"Skipping {output_filename}: Invalid file list")
                continue

            # Create full output path
            output_path = os.path.join(base_output_dir, output_filename)

            # Reference pressure and epsilon for calculations
            P_REF = 1e-6  # reference pressure
            EPSILON = np.finfo(float).eps

            # Containers for combined results
            combined_f = []
            combined_t = []
            combined_psd_db = []
            combined_bb_spl_db = []

            # Track processed and skipped files
            processed_files = []
            skipped_files = []

            # Process each WAV file
            for i, single_file in enumerate(file_list, 1):
                # Progress tracking
                progress = int((i/len(file_list))*100)
                file_count = f"({i}/{len(file_list)})"
                progress_bar = "[" + "=" * (progress//2) + " " * (50 - progress//2) + "]"
                print(f"\rProcessing {output_filename}: {file_count} {progress_bar} {progress}%", end="", flush=True)
                
                try:
                    # Load audio file
                    audio_object = Audio.from_file(single_file)

                    # Extract timestamp
                    timestamp_str = extract_timestamp_from_filename(os.path.basename(single_file))
                    start_time = pytz.timezone("UTC").localize(timestamp_str)

                    # Calibrate hydrophone signal
                    audio_object = calibrate_hydrophone_signal(audio_object, sensitivity, gain, bit_depth=16)

                    # Apply optional bandpass filter
                    samples = (audio_object.bandpass(low_f=low_f, high_f=high_f, order=4).samples 
                               if low_f is not None and high_f is not None 
                               else audio_object.samples)

                    # Determine sample rate
                    sample_rate = audio_object.sample_rate if fs is None else fs
                    
                    # Compute Welch's PSD
                    nperseg = int(sample_rate * window_length)
                    noverlap = int(nperseg * overlap)

                    f, psd_welch = signal.welch(
                        samples, 
                        fs=sample_rate,
                        window=window, 
                        nperseg=nperseg,
                        noverlap=noverlap, 
                        scaling=scaling
                    )
                    
                    # Convert PSD to dB
                    psd_db = 10 * np.log10(psd_welch / (P_REF**2))
                    
                    # Compute broadband SPL
                    power_total = np.trapz(psd_welch, f)
                    bb_spl_db = 10 * np.log10(power_total / (P_REF**2) + EPSILON)
                    
                    # Collect data for combined dataset
                    combined_f = f  # Frequency array is the same for all files
                    start_time_utc = start_time.astimezone(pytz.UTC).replace(tzinfo=None)
                    combined_t.append(np.datetime64(start_time_utc))

                    combined_psd_db.append(psd_db)
                    combined_bb_spl_db.append(bb_spl_db)
                    
                    processed_files.append(single_file)

                except Exception as e:
                    print(f"\nError processing {single_file}: {e}")
                    skipped_files.append((single_file, str(e)))
                    continue

            # Print processing summary for this file
            print(f"\n\nProcessing Summary for {output_filename}:")
            print(f"Total files: {len(file_list)}")
            print(f"Processed files: {len(processed_files)}")
            print(f"Skipped files: {len(skipped_files)}")
            
            if skipped_files:
                print("\nSkipped Files:")
                for file, error in skipped_files:
                    print(f"- {file}: {error}")

            # Check if any files were processed
            if not processed_files:
                print(f"No files could be processed for {output_filename}. Skipping.")
                continue

            # Create combined xarray Dataset
            combined_ds = xr.Dataset(
                {
                    'psd_db': (['time', 'frequency'], np.array(combined_psd_db)),
                    'bb_spl_db': (['time'], combined_bb_spl_db),
                },
                coords={
                    'frequency': combined_f,
                    'time': np.array(combined_t, dtype='datetime64[ns]')
                },
                attrs={
                    'sensitivity': sensitivity,
                    'gain': gain,
                    'sample_rate': sample_rate,
                    'window': window,
                    'window_length': window_length,
                    'overlap': overlap,
                    'scaling': scaling,
                    'low_f': low_f if low_f is not None else 'None',
                    'high_f': high_f if high_f is not None else 'None',
                    'processed_files': processed_files,
                    'skipped_files': [f[0] for f in skipped_files]
                }
            )
            
            # Save combined dataset
            combined_ds.to_netcdf(output_path)
            print(f"\nCombined dataset saved to {output_path}")

            # Store the dataset in the output dictionary
            output_datasets[output_filename] = combined_ds

        return output_datasets

    except Exception as e:
        print(f"Unexpected error in processing: {e}")
        return None

    finally:
        # Always return to the original directory
        os.chdir(original_dir)


def chunk_path(DATA_PATH, OUTPUT_PATH=None, time_segment="1D"):
    """
    Group files based on a specified time window.
    
    Parameters:
    -----------
    DATA_PATH : str
        Input directory path containing files to be grouped
    OUTPUT_PATH : str, optional
        Output directory path. If None, uses DATA_PATH
    time_segment : str, optional
        Time window for grouping
    
    Returns:
    --------
    dict
        A dictionary with time-based groups and their full file paths
    """
    # Use DATA_PATH as OUTPUT_PATH if not specified
    if OUTPUT_PATH is None:
        OUTPUT_PATH = DATA_PATH
    
    # Validate input path
    if not os.path.exists(DATA_PATH):
        raise ValueError(f"The specified input path does not exist: {DATA_PATH}")
    
    # Ensure output path exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Get all files in the directory
    all_files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    
    # Extract timestamps for all files
    file_timestamps = {}
    for filename in all_files:
        try:
            timestamp_str = extract_timestamp_from_filename(filename)
            start_time = pytz.timezone("UTC").localize(timestamp_str)
            file_timestamps[filename] = start_time
        except Exception as e:
            print(f"Skipping file {filename} due to timestamp extraction error: {e}")
    
    # Determine time group function
    def get_time_group(dt):
        if time_segment == '1D':
            return dt.strftime('%Y-%m-%d')
        elif time_segment == '1H':
            return dt.strftime('%Y-%m-%d_%H')
        elif time_segment == '10M':
            # Round down to nearest 10-minute interval
            rounded_minute = (dt.minute // 10) * 10
            return dt.replace(minute=rounded_minute, second=0, microsecond=0).strftime('%Y-%m-%d_%H-%M')
        elif time_segment == '1M':
            return dt.strftime('%Y-%m')
        elif time_segment == '1W':
            # Get the start of the week (Monday)
            return (dt - timedelta(days=dt.weekday())).strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Unsupported time segment: {time_segment}")
    
    # Group files by time window
    grouped_files = {}
    for filename, timestamp in file_timestamps.items():
        time_group = get_time_group(timestamp)
        
        # Create full input and output paths
        input_file_path = os.path.join(DATA_PATH, filename)
        output_file_path = os.path.join(OUTPUT_PATH, filename)
        
        if time_group not in grouped_files:
            grouped_files[time_group] = []
        
        grouped_files[time_group].append(output_file_path)
    
    # Print only the group counts
    print(f"\nFile Grouping by {time_segment} segments:")
    print("-" * 50)
    for time_group, files in sorted(grouped_files.items()):
        print(f"Time Group: {time_group}, Total files: {len(files)}")
    
    return grouped_files

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
    rms_psd_linear = np.sqrt(np.mean(psd_linear**2, axis=0))
    
    # Convert back to dB
    rms_psd_db = 10 * np.log10(rms_psd_linear)
    
    # 2. Compute percentiles (unchanged, correct)
    percentile_levels = [1, 5, 50, 95, 99]
    percentiles_db = {
        f"{p}%": np.percentile(psd_db_array, p, axis=0)
        for p in percentile_levels
    }
    
    return f, rms_psd_db, percentiles_db


def compute_bb_spl(ds):
    """
    Compute broadband SPL (dB re 1 μPa) from PSD in dB.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with dimensions (time, frequency), containing:
        - psd_db (dB) : Power Spectral Density
        - frequency (Hz) : Frequency bins
        - sensitivity (dB) : Sensor sensitivity (optional, from attributes)
    
    Returns
    -------
    xarray.Dataset
        With variables:
        - time : Time coordinates from input
        - bb_spl : Computed broadband SPL (dB re 1 μPa)
        - site : Site information from input dataset
    """
    # Frequency resolution
    df = float(ds.frequency[1] - ds.frequency[0])  # Hz
    
    # Convert PSD from dB to linear units
    psd_db = ds.psd_db
    
    psd_linear = 10 ** (psd_db / 10)  # Convert to μPa²/Hz
    
    # Integrate over frequency and convert to dB
    p_total = psd_linear.sum(dim='frequency') * df
    bb_spl = 10 * np.log10(p_total)  # dB re 1 μPa
    
    # Return as xarray with time coordinate and site info
    result = xr.Dataset(
        data_vars={
            'bb_spl': (('time'), bb_spl.data),
        },
        coords={'time': ds.time}
    )
    
    # Preserve site information if it exists
    if 'site' in ds.variables:
        result['site'] = ds.site
    elif 'site' in ds.attrs:
        result.attrs['site'] = ds.attrs['site']
    
    # Copy other relevant attributes
    for attr in ds.attrs:
        if attr not in result.attrs:
            result.attrs[attr] = ds.attrs[attr]
    
    return result


def segment_bb_spl(ds, segment_duration='30min'):
    '''
    segment_duration : str, e.g. '30min', '1h', '7D','1W','1M'
    '''
    # Ensure input is a DataArray of broadband SPL
    if 'bb_spl' not in ds:
        raise ValueError("Input dataset must contain 'bb_spl' variable")

    # Extract broadband SPL as DataArray
    bb_spl = ds.bb_spl

    # Perform time resampling
    grouped = bb_spl.resample(time=segment_duration)

    # Compute percentiles WITHIN EACH SEGMENT in dB domain
    percentiles = np.array([
        np.percentile(group[1].values, [1, 5, 50, 95, 99])
        for group in grouped
    ])

    # RMS calculation (in dB domain)
    bb_spl_linear = 10 ** (bb_spl / 10)
    grouped_linear = bb_spl_linear.resample(time=segment_duration)
    
    # RMS in linear space, convert back to dB
    rms_bb_spl_linear = np.sqrt(grouped_linear.mean()**2)
    rms_bb_spl = 10 * np.log10(rms_bb_spl_linear)

    return (
        rms_bb_spl.time.values,  # Segment center times
        rms_bb_spl.values,       # RMS SPL values (in dB)
        percentiles              # Percentiles [n_segments × 5]
    )

def segment_bb_spl_median(ds, segment_duration='30min'):
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


def plot_psd_lineplot(x, y, percentiles=None, xscale='log', yscale='linear',
             width=8, height=4, title='PSD', 
             grid=True, xlim=None, ylim=None, save=False, filename='spectrum_line_plot.png',
             dpi=300, colors=None, xlabel=None, ylabel=None, show_legend=True, ax=None):
    """
    Plot PSD with optional percentiles and legend control.
    
    Args:
        x: X-axis data (frequency or time)
        y: Y-axis data (PSD or SPL values)
        percentiles: Dictionary of percentile data
        xscale: X-axis scale ('log' or 'linear')
        yscale: Y-axis scale ('log' or 'linear')
        width: Figure width
        height: Figure height
        title: Plot title
        grid: Whether to show grid
        xlim: X-axis limits tuple (min, max)
        ylim: Y-axis limits tuple (min, max)
        save: Whether to save figure
        filename: Filename to save plot
        dpi: Resolution for saved plot
        colors: Dictionary mapping percentile labels to colors
        xlabel: X-axis label
        ylabel: Y-axis label
        show_legend: Whether to show the legend (True or False)
        ax: Existing axes to plot on (if None, create new figure)
    
    Returns:
        matplotlib.axes.Axes: The plot's axes object
    """
    default_colors = {
        "1%": "lightblue",
        "5%": "skyblue",
        "50%": "blue",
        "95%": "darkblue",
        "99%": "navy"
    }
    colors = colors if colors is not None else default_colors
    
    xlabel = xlabel if xlabel is not None else ('Frequency (Hz)' if percentiles else 'Time (s)')
    ylabel = ylabel if ylabel is not None else ('PSD (dB re 1 µPa²/Hz)' if percentiles else 'Broadband SPL (dB re 1 µPa)')
    
    # Create figure if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
    
    # Plot data
    ax.plot(x, y, label='RMS level', color='red', linestyle='--', linewidth=2)
    
    if percentiles:
        for label, values in percentiles.items():
            color = colors.get(label, 'gray')
            ax.plot(x, values, '-', label=f'{label} percentile', color=color, alpha=0.7)
    
    # Set scales and limits
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Set labels and grid
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    
    # Handle legend based on show_legend parameter
    if show_legend:
        ax.legend(loc='best')
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Apply tight layout if creating a new figure
    if ax is None:
        plt.tight_layout()
    
    # Save if requested
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    # Only show if creating a new figure
    if ax is None:
        plt.show()
    
    return ax


def plot_bb_spl_lineplot(times, rms_spl, percentiles=None,
                width=8, height=4,
                title='Broadband SPL Segmentation',
                grid=True, ylim=None,
                save=False, xlabel='', ylabel='SPL (dB re 1 μPa)',
                filename='broadband_spl_plot.png',
                dpi=300,
                show_legend=True,
                fixed_width=True):
    """
    Plot segmented broadband Sound Pressure Level (SPL) data.
    
    Parameters
    ----------
    times : array-like
        Array of segment center times (datetime64 or datetime objects)
    rms_spl : array-like
        RMS SPL values for each segment
    percentiles : array-like, optional
        Percentile values with shape (n_segments, 5)
        Columns should correspond to [1%, 5%, 50%, 95%, 99%] percentiles
    width : float, optional
        Figure width in inches (default: 8)
    height : float, optional
        Figure height in inches (default: 4)
    title : str, optional
        Plot title (default: 'Broadband SPL Segmentation')
    grid : bool, optional
        Whether to show grid lines (default: True)
    ylim : tuple, optional
        Y-axis limits (default: None)
    save : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saved plot (default: 'broadband_spl_plot.png')
    dpi : int, optional
        Resolution of saved plot (default: 300)
    show_legend : bool, optional
        Whether to show the legend (default: True)
    fixed_width : bool, optional
        Whether to maintain consistent width regardless of x-axis length (default: True)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """

    # Create figure and axis
    if fixed_width:
        # Use constrained layout for better automatic spacing
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot RMS Level as a red line
    ax.plot(times, rms_spl, label='RMS level', color='red', linewidth=2)
    
    # Plot percentiles as shaded area if provided
    if percentiles is not None:
        # Get 5% and 95% percentiles (indices 1 and 3)
        lower_percentile = percentiles[:, 1]  # 5%
        upper_percentile = percentiles[:, 3]  # 95%
        
        # Create shaded area between 5% and 95% percentiles
        ax.fill_between(times, lower_percentile, upper_percentile, 
                        color='gray', alpha=0.3, label='5-95% percentile')
    
    # Format x-axis to show only year and month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Get all unique months in the data
    try:
        import pandas as pd
        pd_times = pd.to_datetime(times)
        month_starts = pd.Series(pd_times.strftime('%Y-%m')).unique()
        x_labels = month_starts
    except ImportError:
        month_strings = np.array([np.datetime_as_string(t, unit='M') for t in times])
        x_labels = np.unique(month_strings)

    # Set x-ticks and rotate them
    plt.xticks(x_labels, rotation=45, ha='left')
    ax.tick_params(axis='x', pad=0)     

    # Add vertical lines to separate months
    try:
        for i in range(1, len(month_starts)):
            month_dt = pd.to_datetime(month_starts[i])
            ax.axvline(month_dt, color='gray', linestyle=':', alpha=0.7)
    except NameError:
        for month_str in np.unique(month_strings)[1:]:
            month_dt = np.datetime64(month_str)
            ax.axvline(month_dt, color='gray', linestyle=':', alpha=0.7)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid and legend
    if grid:
        ax.grid(linestyle=':', alpha=0.5)
    
    if show_legend:
        ax.legend(loc='lower right')
    
    if not fixed_width:
        plt.tight_layout()
    
    # Save plot if requested
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return None


def plot_bb_spl_boxplot(ds, segment_duration='30min', width=8, height=4,
                   title='Broadband SPL Box Plot', grid=True, xlim=None, ylim=None,
                   save=False, filename='bb_spl_box_plot.png', dpi=300,
                   xlabel=None, ylabel=None, showfliers=True, color='skyblue'):
    # Convert to linear space and resample
    bb_spl_linear = 10 ** (ds.bb_spl / 10)
    grouped = bb_spl_linear.resample(time=segment_duration)

    # Prepare data and labels
    data_to_plot = []
    bin_edges = []
    for segment_start, group in grouped:
        if len(group) > 0:
            data_to_plot.append(10 * np.log10(group))
            bin_edges.append(pd.to_datetime(segment_start))

    if not data_to_plot:
        raise ValueError("No data available for the specified time segments")

    # Create figure with custom size
    fig, ax = plt.subplots(figsize=(width, height))

    # Improved box positioning
    # Use explicit width to prevent overlap
    box_width = 0.1  # Adjust this value to control spacing
    positions = np.arange(len(data_to_plot))

    # Create boxplot with improved positioning
    bp = ax.boxplot(data_to_plot, positions=positions, 
                    patch_artist=True,
                    showfliers=showfliers, 
                    showmeans=False, 
                    showcaps=True,
                    widths=box_width)

    # Customize boxes
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=0.7)
    for median in bp['medians']:
        median.set(color='red', linewidth=1)

    # Format x-axis with actual time labels
    ax.set_xticks(positions)
    ax.set_xticklabels([edge.strftime('%Y-%m') for edge in bin_edges], 
                       rotation=45, ha='right')
    # ax.set_xticklabels([edge.strftime('%Y-%m-%d %H:%M') for edge in bin_edges], 
    #                    rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel(xlabel if xlabel else f'Time ({segment_duration})')
    ax.set_ylabel(ylabel if ylabel else 'Broadband SPL (dB re 1 μPa)')
    ax.set_title(title)

    # Set grid
    ax.grid(grid, linestyle=':', alpha=0.7)

    # Set axis limits if specified
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()

    # Save figure if requested
    if save:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    plt.show()
    return None

def plot_grouped_bb_spl_lineplot(times_list, rms_spl_list, percentiles=None,
                width=8, height=4,
                title='Broadband SPL Segmentation',
                grid=True, ylim=None,
                save=False, xlabel='', ylabel='Broadband SPL (dB re 1 μPa)',
                filename='broadband_spl_plot.png',
                dpi=300,
                show_legend=True,
                fixed_width=True,
                labels=None,
                colors=None,
                linestyles=None):
    """
    Plot multiple broadband Sound Pressure Level (SPL) data series.
    
    Parameters
    ----------
    times_list : list of array-like
        List of arrays containing segment center times (datetime64 or datetime objects)
    rms_spl_list : list of array-like
        List of arrays containing RMS SPL values for each segment
    percentiles : list of array-like, optional
        List of percentile values with shape (n_segments, 5) for each series
        Columns should correspond to [1%, 5%, 50%, 95%, 99%] percentiles
        Set to None to skip plotting percentiles
    width : float, optional
        Figure width in inches (default: 8)
    height : float, optional
        Figure height in inches (default: 4)
    title : str, optional
        Plot title (default: 'Broadband SPL Segmentation')
    grid : bool, optional
        Whether to show grid lines (default: True)
    ylim : tuple, optional
        Y-axis limits (default: None)
    save : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saved plot (default: 'broadband_spl_plot.png')
    dpi : int, optional
        Resolution of saved plot (default: 300)
    show_legend : bool, optional
        Whether to show the legend (default: True)
    fixed_width : bool, optional
        Whether to maintain consistent width regardless of x-axis length (default: True)
    labels : list of str, optional
        Labels for each data series (default: None, will use 'Series 1', 'Series 2', etc.)
    colors : list of str, optional
        Colors for each data series (default: None, will use default color cycle)
    linestyles : list of str, optional
        Line styles for each data series (default: None, will use solid lines)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Handle single time series case by converting to lists
    if not isinstance(times_list[0], (list, tuple, np.ndarray)) or (
            hasattr(times_list[0], 'shape') and len(times_list[0].shape) == 0):
        times_list = [times_list]
        rms_spl_list = [rms_spl_list]
        if percentiles is not None and not isinstance(percentiles[0], (list, tuple, np.ndarray)):
            percentiles = [percentiles]
    
    # Set default labels if not provided
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(times_list))]
    
    # Set default colors if not provided
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Set default linestyles if not provided
    if linestyles is None:
        linestyles = ['-'] * len(times_list)
    
    # Create figure and axis
    if fixed_width:
        # Use constrained layout for better automatic spacing
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(width, height))
    
    # Collect all times to determine overall x-axis range
    all_times = []
    for times in times_list:
        all_times.extend(times)
    
    # Plot each data series
    for i, (times, rms_spl) in enumerate(zip(times_list, rms_spl_list)):
        color_idx = i % len(colors)
        line_idx = i % len(linestyles)
        
        # Plot RMS Level
        ax.plot(times, rms_spl, 
                label=labels[i], 
                color=colors[color_idx], 
                linestyle=linestyles[line_idx],
                linewidth=2)
        
        # Plot percentiles as shaded area if provided
        if percentiles is not None and i < len(percentiles) and percentiles[i] is not None:
            # Get 5% and 95% percentiles (indices 1 and 3)
            lower_percentile = percentiles[i][:, 1]  # 5%
            upper_percentile = percentiles[i][:, 3]  # 95%
            
            # Create shaded area between 5% and 95% percentiles
            ax.fill_between(times, lower_percentile, upper_percentile, 
                            color=colors[color_idx], alpha=0.3, 
                            label=f'{labels[i]} 5-95% percentile')
    
    # Format x-axis to show only year and month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Get all unique months in the data
    try:
        import pandas as pd
        pd_times = pd.to_datetime(all_times)
        month_starts = pd.Series(pd_times.strftime('%Y-%m')).unique()
        x_labels = [pd.to_datetime(month) for month in month_starts]
    except ImportError:
        month_strings = np.array([np.datetime_as_string(t, unit='M') for t in all_times])
        unique_months = np.unique(month_strings)
        x_labels = [np.datetime64(month) for month in unique_months]

    # Set x-ticks and rotate them
    plt.xticks(x_labels, rotation=45, ha='left')
    ax.tick_params(axis='x', pad=0)     

    # Add vertical lines to separate months
    try:
        for month_dt in x_labels[1:]:  # Skip the first month
            ax.axvline(month_dt, color='gray', linestyle=':', alpha=0.7)
    except Exception:
        pass  # Skip if there's an issue with the vertical lines
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid and legend
    if grid:
        ax.grid(linestyle=':', alpha=0.5)
    
    if show_legend:
        ax.legend(loc='best')
    
    if not fixed_width:
        plt.tight_layout()
    
    # Save plot if requested
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    return None

def plot_grouped_bb_spl_boxplot(
    data,                      # Input data (xarray Dataset or DataFrame)
    segment_duration='1M',      # Time segment: '1M' (monthly), '30min', '1D', etc.
    width=8,                   # Plot width (inches)
    height=4,                  # Plot height (inches)
    title='Broadband SPL Box Plot',  # Plot title
    grid=True,                 # Show grid?
    xlim=None,                 # X-axis limits (e.g., [start, end])
    ylim=None,                 # Y-axis limits (e.g., [min, max])
    save=False,                # Save plot?
    filename='bb_spl_box_plot.png',  # Output filename
    xlabel=None,               # Custom x-axis label
    ylabel=None,               # Custom y-axis label
    showfliers=True,           # Show outliers?
    color='RdBu',           # Box color (single color or palette name)
    hue_order=None,            # Order of sites (e.g., ['Site1A', 'Site3A', 'Site4A'])
    bbox_to_anchor=(1, 1),
    loc='upper right'          # Legend position
):
    """
    Plots a grouped boxplot of broadband SPL by site and time segment.
    
    Parameters:
    -----------
    data : xarray.Dataset or pd.DataFrame
        Input data with 'bb_spl', 'time', and 'site' dimensions.
    segment_duration : str
        Pandas-compatible time frequency (e.g., '1M', '30min', '1D').
    width, height : float
        Plot dimensions in inches.
    title, xlabel, ylabel : str
        Plot labels.
    grid, showfliers : bool
        Toggle grid/outliers.
    xlim, ylim : list
        Axis limits.
    save : bool
        Save the plot to file.
    filename : str
        Output filename (if save=True).
    color : str
    """
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        df = data['bb_spl'].to_dataframe().reset_index()
    else:
        df = data.copy()
    
    # Ensure 'time' is datetime
    if not is_datetime(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    
    # Drop NaNs
    df = df.dropna(subset=['bb_spl'])
    
    # Segment time into bins
    if segment_duration.endswith('min'):
        freq = f'{int(segment_duration[:-3])}T'  # '30min' → '30T'
    else:
        freq = segment_duration
    
    df['time_segment'] = df['time'].dt.to_period(freq).astype(str)
    
    # Set up plot
    plt.figure(figsize=(width, height))
    
    # Create boxplot
    sns.boxplot(
        data=df,
        x='time_segment',
        y='bb_spl',
        hue='site',
        hue_order=hue_order,
        palette=color if isinstance(color, (str, dict, list)) else None,
        showfliers=showfliers,
        width=0.6,
    )
    
    # Customize plot
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel if xlabel else 'Time Segment (' + segment_duration + ')', fontsize=12)
    plt.ylabel(ylabel if ylabel else 'BB SPL (dB)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='', bbox_to_anchor=bbox_to_anchor, loc=loc)
    
    if grid:
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()


def compute_toctave_band(center_f):
    """
    Apply a 1/3-octave bandpass filter based on the center frequency
    """
    # Calculate the lower and upper frequency limits for the 1/3-octave band
    f_low = center_f * 2**(-1/6)
    f_high = center_f * 2**(1/6)
 
    return f_low, f_high

