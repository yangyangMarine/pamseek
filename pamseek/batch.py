import os
import glob
import numpy as np
from opensoundscape import Audio, audio
import opensoundscape
import xarray as xr
import pandas as pd

import scipy.signal as signal

# for auto load time_stamp on the loaded files
from datetime import datetime
import pytz


def compute_single_audio_file(path):
    """
    Reads all .wav files from a directory, processes each file individually,
    applies calibration, computes spectral metrics, and saves the results as .npy files.
    Prints progress as a percentage.

    Parameters:
    -----------
    path : str
        Path to the directory containing .wav files

    Returns:
    --------
    None
    """
    # Set working directory
    original_dir = os.getcwd()
    os.chdir(path)
    print(f"Reading audio files from: {os.getcwd()}")

    # List all .wav files
    wav_files = glob.glob('*.wav')
    print(f"Found {len(wav_files)} .wav files in the directory:")
    for file in wav_files:
        print(f"- {file}")

    if len(wav_files) == 0:
        os.chdir(original_dir)  # Return to original directory
        raise ValueError(f"No .wav files found in {path}")

    # Sort wav files to ensure chronological order (optional, but useful for consistency)
    wav_files.sort()

    # Loop through each file and process it
    total_files = len(wav_files)
    for i, single_file in enumerate(wav_files):
        # Calculate progress percentage
        progress = ((i + 1) / total_files) * 100
        print(f"\n=============================== {progress:.0f}% finished ===============================")
        print(f"Processing file: {single_file}")

        # Load the audio object
        audio_object = Audio.from_file(single_file)

        # Extract timestamp from filename
        timestamp_str = extract_timestamp_from_filename(single_file)
        start_time = pytz.timezone("UTC").localize(timestamp_str)
        audio_object.metadata['recording_start_time'] = start_time

        # Step 1: Apply hydrophone calibration
        audio_object = cal_hydrophone(audio_object, -156, gain=0, bit_depth=24)

        # Step 2: Compute compute_PSD
        f, t, psd_db = compute_PSD(
            audio_object,
            fs=None,
            window='hann',
            window_length=1.0,  # 1 sec
            overlap=0.5,  # 50%
            scaling='density'
        )
        
        # Step 3: Save the results as a .npy file
        output_filename = os.path.splitext(single_file)[0] + ".npy"  # Replace .wav with .npy
        np.save(output_filename, {
            'f': f,
            't': t,
            'psd_db': psd_db,
            'start_time': start_time.isoformat()  # Save start_time as ISO format string
        })
        print(f"Spectral metrics saved to: {output_filename}")
        
        # save to NETcdf or Zarr format
        # # Transpose psd_db to align dimensions with time and frequency
        # psd_db = psd_db.T  # Transpose to shape (3599, 48001)

        # # Step 3: Save the results as a .zarr file using xarray
        # output_filename = os.path.splitext(single_file)[0] + ".zarr"  # Replace .wav with .zarr

        # # Create an xarray Dataset
        # ds = xr.Dataset(
        #     {
        #         'psd_db': (['time', 'frequency'], psd_db),  # Power spectral density
        #     },
        #     coords={
        #         'time': t,  # Time vector
        #         'frequency': f,  # Frequency vector
        #     },
        #     attrs={
        #         'start_time': start_time.isoformat(),  # Start time as an attribute
        #         'file_name': single_file,  # Original file name
        #         'description': 'Spectral metrics computed from audio file',
        #     }
        # )

        # # Save to NetCDF file
        # ds.to_netcdf(output_filename)
        # print(f"Spectral metrics saved to: {output_filename}")  
        
    # Return to original directory
    os.chdir(original_dir)

def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename in format '20240717T093000_...' for loggerhead hydrophone
    
    Parameters:
    -----------
    filename : str
        Filename to extract timestamp from
        
    Returns:
    --------
    datetime
        Datetime object representing the timestamp
    """
    # Extract the timestamp part before the first underscore
    timestamp_part = filename.split('_')[0]
    
    # Parse the timestamp string into a datetime object
    # Format: YYYYMMDDTHHMMSS
    try:
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

def cal_hydrophone(audio_data, sensitivity_dB, gain=0, bit_depth=24):
    """
    Corrects a raw hydrophone audio signal by applying sensitivity, gain, and accounting for bit depth.
    
    Parameters:
    -----------
    audio_data : opensoundscape.Audio or numpy.ndarray
        Audio object or 1-dimensional array of raw digital wave signal
    sensitivity_dB : float
        Hydrophone sensitivity in dB re 1V/µPa
    gain : float, optional
        Additional gain applied to the signal in dB (default: 0 dB)
    bit_depth : int, optional
        Bit depth of the audio signal (default: 24)
        
    Returns:
    --------
    opensoundscape.Audio or numpy.ndarray
        Audio object or array with corrected signal in Pascals (Pa)
    """
    # Handle either Audio object or numpy array
    is_audio_object = hasattr(audio_data, 'samples')
    samples = audio_data.samples if is_audio_object else audio_data
    
    # Step 1: Normalize raw data based on bit depth
    if bit_depth == 16:
        normalized_signal = samples / 32768.0  # For 16-bit signed integer data
    elif bit_depth == 24:
        normalized_signal = samples / 8388608.0  # For 24-bit signed integer data
    elif bit_depth == 32:
        if np.issubdtype(samples.dtype, np.integer):
            normalized_signal = samples / 2147483648.0  # For 32-bit signed integer data
        else:
            normalized_signal = samples  # Assuming it's already float32 normalized to [-1, 1]
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    # Optional: Remove DC offset
    # normalized_signal = normalized_signal - np.mean(normalized_signal)
    
    # Step 2: Convert sensitivity from dB to linear scale
    # Sensitivity in dB re 1V/µPa
    sensitivity_linear = 10**(sensitivity_dB/20)  
    
    # Convert sensitivity to Pa/V (reciprocal of V/Pa)
    receive_sensitivity = 1 / sensitivity_linear  # Pa/V
    
    # Step 3: Apply sensitivity and gain to convert voltage to pressure (Pa)
    gain_linear = 10**(gain/20)
    sound_pressure = normalized_signal * receive_sensitivity * gain_linear
    
    # Return the appropriate type
    if is_audio_object:
        audio_data.samples = sound_pressure
        return audio_data
    else:
        return sound_pressure

def compute_PSD(audio_data, fs=None, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Computes the spectral metrics of audio data
    
    Parameters:
    -----------
    audio_data : opensoundscape.Audio or numpy.ndarray
        Audio object or 1-dimensional array of audio samples in Pascals
    fs : float, optional
        Sampling rate in Hz. Required if audio_data is numpy array.
    window : str or tuple, optional
        Window function to use (default: 'hann')
    window_length : float, optional
        Length of each window in seconds (default: 1.0)
    overlap : float, optional
        Overlap factor between 0 and 1 (default: 0.5 for 50% overlap)
    scaling : str, optional
        Scaling mode for the spectrogram ('density' or 'spectrum', default: 'density')
    
    Returns:
    --------
    tuple
        (f, t, Sxx_db, rms_level, percentiles, spl, rms_spl)
        - f: frequencies array
        - t: time array
        - Sxx_db: spectrogram in dB re 1 µPa²/Hz
        - rms_level: RMS levels for each frequency bin in dB
        - percentiles: dictionary of percentile levels (1%, 5%, 50%, 95%, 99%) for each frequency bin
    """
    # Handle input data
    if hasattr(audio_data, 'samples'):
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate
    else:
        samples = audio_data
        if fs is None:
            raise ValueError("Sample rate (fs) must be provided when input is not an Audio object")
        sample_rate = fs
    
    # Calculate segment parameters
    nperseg = int(sample_rate * window_length)
    noverlap = int(nperseg * overlap)
    
    # Calculate spectrogram
    f, t, psd = signal.spectrogram(
        samples, 
        fs=sample_rate,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap, 
        scaling=scaling
    )
    
    # Reference pressure for underwater acoustics
    p_ref = 1e-6  # 1 µPa in Pa
    
    # Convert spectrogram to dB re 1 µPa²/Hz
    # Avoid log of zero by adding small constant
    epsilon = np.finfo(float).eps
    psd_db = 10 * np.log10(psd / (p_ref**2) + epsilon)
    

    return f, t, psd_db