import numpy as np
import scipy.signal as signal

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
        (f, Pxx_db) where f is frequencies array and Pxx_db is PSD in dB re 1 µPa²/Hz
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
    f, Pxx = signal.welch(
        samples, 
        fs=sample_rate,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap, 
        scaling=scaling
    )
    
    # Convert PSD from Pa²/Hz to dB re 1 µPa²/Hz
    P_ref = 1e-6  # Reference pressure for underwater (1 µPa in Pascals)
    Pxx_db = 10 * np.log10(Pxx / (P_ref**2))
    
    return f, Pxx_db


def compute_spectrogram(audio_data, fs=None, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Computes the spectrogram of audio data and related statistics.
    
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
        Scaling mode for the spectrogram ('density' or 'spectrum', default: 'density')
    
    Returns:
    --------
    tuple
        (f, t, Sxx_db, rms_level, percentiles)
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
    f, t, Sxx = signal.spectrogram(
        samples, 
        fs=sample_rate,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap, 
        scaling=scaling
    )
    
    # Reference pressure for underwater acoustics
    P_ref = 1e-6  # 1 µPa in Pa
    
    # Compute SPL for each time bin (over frequency, axis=0)
    spl = 10 * np.log10(np.sum(Sxx * (f[1] - f[0]), axis=0) / (p_ref**2))

    # Compute rmsSPL
    spl_rms = 10 * np.log10(np.mean(10**(spl / 10)))
    
    # Convert spectrogram to dB re 1 µPa²/Hz
    # Avoid log of zero by adding small constant
    epsilon = np.finfo(float).eps
    Sxx_db = 10 * np.log10(Sxx / (P_ref**2) + epsilon)
    
    # Calculate RMS levels for each frequency bin
    # Work with linear values first, then convert to dB, integrating over time (axis=1), freuqency (axis=0)
    psd_rms = 10 * np.log10(np.mean(Sxx, axis=1) / (P_ref**2) + epsilon)
    
    # Calculate percentiles for each frequency bin
    percentiles = {
        "1%": np.percentile(Sxx_db, 1, axis=1),
        "5%": np.percentile(Sxx_db, 5, axis=1),
        "50%": np.percentile(Sxx_db, 50, axis=1),
        "95%": np.percentile(Sxx_db, 95, axis=1),
        "99%": np.percentile(Sxx_db, 99, axis=1)
    }
    
    return f, t, Sxx_db, psd_rms, percentiles, spl, spl_rms
