import os
import glob
import numpy as np
from opensoundscape import Audio, audio
import opensoundscape

# for auto load time_stamp on the loaded files
from datetime import datetime
import pytz


def load_audio_files(path):
    """
    Reads all .wav files from a directory and concatenates them into a single Audio object.
    Each file's timestamp is extracted from its filename.
    
    Parameters:
    -----------
    path : str
        Path to the directory containing .wav files
        
    Returns:
    --------
    opensoundscape.Audio
        Concatenated Audio object with metadata
    """
    # Set working directory
    original_dir = os.getcwd()
    os.chdir(path)
    print(f"Reading audio files from: {os.getcwd()}")
    
    # List all .wav files
    wav_files = glob.glob('*.wav')
    print(f"Found {len(wav_files)} .wav files:")
    for file in wav_files[:5]:  # Show first 5 files
        print(f"- {file}")
        
    if len(wav_files) == 0:
        os.chdir(original_dir)  # Return to original directory
        raise ValueError(f"No .wav files found in {path}")
    
    # Sort wav files to ensure chronological order
    wav_files.sort()
    
    # Initialize with the first audio object
    first_file = wav_files[0]
    audio_object = Audio.from_file(first_file)
    
    # Extract timestamp from filename for the first file
    timestamp_str = extract_timestamp_from_filename(first_file)
    start_time = pytz.timezone("UTC").localize(timestamp_str)
    audio_object.metadata['recording_start_time'] = start_time
    
    # If there are more files, load and concatenate them
    if len(wav_files) > 1:
        total_files = len(wav_files) - 1  # Subtract 1 as already loaded the first file
        for i, file in enumerate(wav_files[1:]):
            temp = Audio.from_file(file)
            
            # Extract timestamp from filename for this file
            timestamp_str = extract_timestamp_from_filename(file)
            file_start_time = pytz.timezone("UTC").localize(timestamp_str)
            temp.metadata['recording_start_time'] = file_start_time
            
            # Concatenate with existing audio
            audio_object = opensoundscape.audio.concat([audio_object, temp])
            
            # Print progress
            percentage_complete = ((i + 1) / total_files) * 100 
            print(f"Concatenation finished --------------- {round(percentage_complete)}%")
    
    # Return to original directory
    os.chdir(original_dir)
    
    return audio_object

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