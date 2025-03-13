import numpy as np
import pandas as pd
import scipy.signal as signal


def compute_spectral_stats(f=None, t=None, rms_psd=None, rms_spl=None):
    # Function to compute statistics of the given variables and print in a table format
    
    # List of 1D arrays (check if they are not None)
    arrays = {
        'Frequency (f)': f,
        'Time (t)': t,
        'PSD RMS': rms_psd,
        'RMS SPL': rms_spl
    }

    stats = {}

    # Iterate through each variable
    for label, arr in arrays.items():
        if arr is not None:  # If the array is not None, calculate stats
            stats[label] = {
                'Min': np.min(arr),
                'Max': np.max(arr),
                'Mean': np.mean(arr),
                'Std Dev': np.std(arr),
                'Size': arr.size
            }
        else:
            stats[label] = 'Missing data'

    # Convert stats to a pandas DataFrame for pretty printing
    stats_df = pd.DataFrame(stats)

    # Print the stats table
    print(stats_df)


def compute_psd(audio_data, fs=None, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Computes the Power Spectral Density (PSD) of audio data using Welch's method.
    
    Parameters:
    -----------
    audio_data : opensoundscape.Audio or numpy.ndarray
        Audio object or 1-dimensional array of audio samples in Pascals
    fs : float, optional
        Sampling frequency in Hz. Required if audio_data is numpy array.
    window : str or tuple, optional
        Window function to use (default: 'hann')
    window_length : float, optional
        Length of each window in seconds (default: 1.0)
    overlap : float, optional
        Overlap factor between 0 and 1 (default: 0.5 for 50% overlap)
    scaling : str, optional
        Scaling mode for the PSD ('density' or 'spectrum', default: 'density')
    
    Returns:
    --------
    tuple
        (f, psd_db) where f is frequencies array and psd_db is PSD in dB re 1 µPa²/Hz
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
    
    # Calculate PSD using Welch's method
    f, psd = signal.welch(
        samples, 
        fs=sample_rate,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap, 
        scaling=scaling
    )
    
    # Convert PSD from Pa²/Hz to dB re 1 µPa²/Hz
    P_ref = 1e-6  # Reference pressure for underwater (1 µPa in Pascals)
    psd_db = 10 * np.log10(psd / (P_ref**2))
    
    return f, psd_db


def compute_spectral_metrics(audio_data, fs=None, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
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
    
    # Compute SPL
    spl = 10 * np.log10(psd / (p_ref**2))

    # Compute RMS SPL, over frequencies)
    rms_spl_t = 10 * np.log10(np.mean(10**(spl / 10), axis=0))
    
    # Convert spectrogram to dB re 1 µPa²/Hz
    # Avoid log of zero by adding small constant
    epsilon = np.finfo(float).eps
    psd_db = 10 * np.log10(psd / (p_ref**2) + epsilon)
    
    # Calculate RMS levels for each frequency bin
    # Work with linear values first, then convert to dB, integrating over time (axis=1), freuqency (axis=0)
    rms_psd_f = 10 * np.log10(np.mean(psd, axis=1) / (p_ref**2) + epsilon)
    
    # Calculate percentiles for each frequency bin
    percentiles = {
        "1%": np.percentile(psd_db, 1, axis=1),
        "5%": np.percentile(psd_db, 5, axis=1),
        "50%": np.percentile(psd_db, 50, axis=1),
        "95%": np.percentile(psd_db, 95, axis=1),
        "99%": np.percentile(psd_db, 99, axis=1)
    }
    
    return f, t, psd_db, rms_psd_f, percentiles, spl, rms_spl_t


def compute_toctave(audio_object, center_f=None, low_f=None, high_f=None, 
                    fs=None, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Compute third-octave band spectrogram and metrics from an audio object.
    
    Parameters:
    -----------
    audio_object : object
        Audio object with sample rate (fs) and data attributes
    center_frequencies : array-like, optional
        Specific third-octave band center frequencies to analyze
        If None, bands will be determined based on low_f and high_f or the full spectrum
    low_f : float, optional
        Lower frequency boundary for analysis
    high_f : float, optional
        Upper frequency boundary for analysis
    TBD...
        
    Returns:
    --------
    f : ndarray
        Center frequencies of third-octave bands
    t : ndarray
        Time points
    psd_db_band : ndarray
        Power spectral density in dB for each band and time point
    rms_psd_band : ndarray
        Time-averaged RMS of PSD for each frequency band (in dB)
    """
    # Apply bandpass filter if frequency boundaries are provided
    if low_f is not None or high_f is not None:
        filtered_audio = audio_object.bandpass(low_f=low_f, high_f=high_f, order=12)
        audio_data = filtered_audio.data
        fs = filtered_audio.fs
    else:
        audio_data = audio_object.data
        fs = audio_object.fs

    # Handle FFT paras
    if hasattr(audio_data, 'samples'):
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate
    else:
        samples = audio_data
        if fs is None:
            raise ValueError("Sample rate (fs) must be provided when input is not an Audio object")
        sample_rate = fs
        
    # Calculate FFT parameters
    nperseg = int(sample_rate * window_length)
    noverlap = int(nperseg * overlap)
    
    # Calculate spectrogram
    f_spec, t, Sxx = signal.spectrogram(
        audio_data, 
        fs,
        window=window, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling=scaling
    )
                        
    # Determine third-octave band center frequencies if not provided
    if center_f is None:
        # Get frequency range from spectrogram
        f_min = f_spec[0]
        f_max = f_spec[-1]
        
        # If low_f and high_f are provided, adjust the range
        if low_f is not None:
            f_min = max(f_min, low_f)
        if high_f is not None:
            f_max = min(f_max, high_f)
        
        # Determine appropriate third-octave bands within this range
        n_min = int(np.floor(3 * np.log2(f_min / 1000)))
        n_max = int(np.ceil(3 * np.log2(f_max / 1000)))
        
        # Generate band centers
        band_indices = np.arange(n_min, n_max + 1)
        center_f = 1000 * 2**(band_indices/3)
        
        # Filter out any centers outside our actual frequency range
        center_f = center_f[(center_f >= f_min) & 
                                               (center_f <= f_max)]
    
    # Initialize arrays for results
    num_bands = len(center_f)
    psd = np.zeros((num_bands, len(t)))
    psd_db_band = np.zeros((num_bands, len(t)))
    rms_psd_band = np.zeros(num_bands)
    
    # Process each third-octave band
    for i, center in enumerate(center_f):
        # Calculate band edges (lower and upper frequencies)
        f_lower = center / 2**(1/6)
        f_upper = center * 2**(1/6)
        
        # Find indices of frequencies within the band
        indices = np.where((f_spec >= f_lower) & (f_spec <= f_upper))[0]
        
        if len(indices) > 0:
            # Calculate PSD for each time step
            for j in range(len(t)):
                if len(indices) == 1:
                    # Only one frequency bin
                    band_width = f_spec[indices[0]] if indices[0] == 0 else (f_spec[indices[0]+1] - f_spec[indices[0]-1])/2
                    psd[i, j] = Sxx[indices[0], j] * band_width
                else:
                    # Multiple bins - use trapezoidal integration
                    psd[i, j] = np.trapz(Sxx[indices, j], f_spec[indices])
            
            # Convert to dB
            psd_db_band[i, :] = 10 * np.log10(psd[i, :] + 1e-10)
            
            # Calculate RMS values across time (in dB)
            # First get the time-average of the linear PSD
            time_avg_psd = np.mean(psd[i, :])
            # Then convert to dB
            rms_psd_band[i] = 10 * np.log10(time_avg_psd + 1e-10)
    
    return center_f, t, psd_db_band, rms_psd_band
