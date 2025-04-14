import os
import glob
import warnings
import soundfile
import numpy as np
import xarray as xr
import pytz
from datetime import datetime
from scipy import signal
from scipy.signal import butter, lfilter
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import shutil
import netCDF4

import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import soundfile as soundfile


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

def calibrate_hydrophone_signal(samples, sensitivity_db, gain=0):
    """
    Correct a raw hydrophone audio signal by applying sensitivity, gain.
    
    Parameters:
    -----------
    samples : Audio or np.ndarray
        Raw audio samples to be calibrated.
    sensitivity_db : float
        Hydrophone sensitivity in dB re 1V/µPa
    gain : float, optional
        Additional gain applied to the signal in dB (default: 0 dB)
    """
    
    # Convert sensitivity from dB to linear
    sensitivity_linear = 10 ** (sensitivity_db / 20)  # V/μPa
    
    # Convert gain from dB to linear
    gain_linear = 10 ** (gain / 20)
    
    sound_pressure = samples / (gain_linear * sensitivity_linear)
    
    return sound_pressure

def bandpass_filter(audio_data, sample_rate, low_f, high_f):
    """Apply bandpass filter to audio data"""
    if low_f is None or high_f is None:
        return audio_data
        
    nyquist = 0.5 * sample_rate
    
    # Validate frequency bounds
    if low_f >= high_f:
        warnings.warn(f"low_f ({low_f} Hz) must be < high_f ({high_f} Hz). Skipping filter.")
        return audio_data
    if high_f > nyquist:
        warnings.warn(f"high_f ({high_f} Hz) exceeds Nyquist frequency ({nyquist} Hz). Skipping filter.")
        return audio_data
        
    low = low_f / nyquist
    high = high_f / nyquist
    
    b, a = butter(5, [low, high], btype='band')
    filtered = lfilter(b, a, audio_data)
    
    # Check for NaN/Inf
    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
        warnings.warn("Bandpass filter returned NaN/Inf values. Returning original signal.")
        return audio_data
        
    return filtered

def process_single_file(file_path, sensitivity, gain, fs, window, 
                      window_length, overlap, scaling, low_f, high_f):
    """Process a single WAV file and return its results"""
    try:
        # Load audio file
        samples, sample_rate = soundfile.read(file_path)
        
        # Extract timestamp
        filename = os.path.basename(file_path)
        timestamp_str = extract_timestamp_from_filename(filename)
        start_time = pytz.timezone("UTC").localize(timestamp_str)
        
        # Calibrate hydrophone signal using your function
        samples = calibrate_hydrophone_signal(samples, sensitivity, gain)
        
        # Apply bandpass filter if frequency limits are specified
        samples = bandpass_filter(samples, sample_rate, low_f, high_f)
        
        # Determine sample rate
        sample_rate = sample_rate if fs is None else fs
        
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
        psd_db = 10 * np.log10(psd_welch)
        
        # Adding epsilon to avoid log of zero
        EPSILON = np.finfo(float).eps
        
        # Broadband SPL Calculation using trapezoidal integration
        power_total_trapz = np.trapz(psd_welch, f)
        
        # Convert to dB re 1 µPa
        bb_spl = 10 * np.log10(power_total_trapz + EPSILON)
        
        # Return results
        start_time_utc = start_time.astimezone(pytz.UTC).replace(tzinfo=None)
        return {
            'success': True,
            'filename': filename,
            'f': f,
            'psd_db': psd_db,
            'bb_spl': bb_spl,
            'timestamp': np.datetime64(start_time_utc)
        }
        
    except Exception as e:
        return {
            'success': False,
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

def process_audio_files(path, sensitivity, gain, fs=None, 
                       window='hann', window_length=1.0, overlap=0.5, 
                       scaling='density', low_f=None, high_f=None, 
                       output_dir=None, output_filename=None, 
                       n_processes=None):
    """
    Process multiple WAV files in a directory to compute Power Spectral Density (PSD)
    using parallel processing for speed.
    
    Parameters:
    -----------
    path : str
        Path to directory containing WAV files
    sensitivity : float
        Hydrophone sensitivity in dB re 1V/µPa
    gain : float
        Gain applied to the signal in dB
    fs : float, optional
        Sample rate to use (overrides file's sample rate if provided)
    window : str, optional
        Window function to use for Welch's method
    window_length : float, optional
        Window length in seconds
    overlap : float, optional
        Overlap fraction between windows (0 to 1)
    scaling : str, optional
        Scaling type for PSD computation ('density' or 'spectrum')
    low_f : float, optional
        Lower frequency bound for bandpass filter
    high_f : float, optional
        Upper frequency bound for bandpass filter
    output_dir : str, optional
        Directory to save the output file (defaults to input path)
    output_filename : str, optional
        Name of the output NetCDF file
    n_processes : int, optional
        Number of parallel processes to use (defaults to CPU count - 1)
    
    Returns:
    --------
    xarray.Dataset
        Combined dataset with PSD and metadata
    """
    # Suppress numpy runtime warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Store original working directory
    original_dir = os.getcwd()
    
    try:
        # Set up paths
        abs_path = os.path.abspath(path)
        output_dir = os.path.abspath(output_dir or path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Find WAV files
        wav_files = sorted(glob.glob(os.path.join(abs_path, '*.wav')))
        if not wav_files:
            raise ValueError(f"No .wav files found in {path}")
            
        # Check output filename
        if not output_filename:
            output_filename = f"combined_psd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
        
        # Create partial function with fixed parameters
        process_func = partial(
            process_single_file,
            sensitivity=sensitivity,
            gain=gain,
            fs=fs,
            window=window,
            window_length=window_length,
            overlap=overlap,
            scaling=scaling,
            low_f=low_f,
            high_f=high_f
        )
        
        print(f"Processing {len(wav_files)} files using {n_processes} processes...")
        
        # Process files in parallel with progress bar
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, wav_files),
                total=len(wav_files),
                desc="Processing audio files"
            ))
        
        # Separate successful and failed results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        # Print processing summary
        print("\nProcessing Summary:")
        print(f"Total files: {len(wav_files)}")
        print(f"Processed files: {len(successful)}")
        print(f"Skipped files: {len(failed)}")
        
        if failed:
            print("\nSkipped Files:")
            for result in failed:
                print(f"- {result['filename']}: {result['error']}")
                
        # Check if any files were processed
        if not successful:
            print("No files could be processed. Returning None.")
            return None
            
        # Get frequency array from first successful result (all should be the same)
        combined_f = successful[0]['f']
        
        # Collect data for combined dataset
        combined_t = np.array([r['timestamp'] for r in successful])
        combined_psd_db = np.array([r['psd_db'] for r in successful])
        combined_bb_spl = np.array([r['bb_spl'] for r in successful])
        
        # Get sample file to determine sample rate if not specified
        if fs is None:
            _, sample_rate = soundfile.read(wav_files[0])
        else:
            sample_rate = fs
        
        # Create combined xarray Dataset
        output_path = os.path.join(output_dir, output_filename)
        combined_ds = xr.Dataset(
            {
                'psd_db': (['time', 'frequency'], combined_psd_db),
                'bb_spl': (['time'], combined_bb_spl)
            },
            coords={
                'frequency': combined_f,
                'time': combined_t
            },
            attrs={
                'sensitivity': sensitivity,
                'gain': gain,
                'sample_rate': sample_rate,
                'window_length': window_length,
                'overlap': overlap,
                'low_f': low_f if low_f is not None else 'None',
                'high_f': high_f if high_f is not None else 'None',
            }
        )
        
        # Save combined dataset with explicit encoding for strings
        encoding = {var: {'dtype': 'str'} for var in combined_ds.variables 
                   if combined_ds[var].dtype.kind == 'U'}
        combined_ds.to_netcdf(output_path, engine='netcdf4', encoding=encoding)
        print(f"Combined dataset saved to {output_path}")
        
        return combined_ds
        
    except Exception as e:
        print(f"Unexpected error in processing: {e}")
        return None
        
    finally:
        # Always restore original working directory
        os.chdir(original_dir)