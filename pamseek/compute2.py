import numpy as np
import pandas as pd
import scipy.signal as signal


def compute_spectral_stats(f=None, t=None, rms_psd=None, rms_spl=None):
    """
    Compute statistics of spectral variables and print them in a table format.
    
    Parameters:
    -----------
    f : ndarray, optional
        Frequency array
    t : ndarray, optional
        Time array
    rms_psd : ndarray, optional
        Root mean square of Power Spectral Density
    rms_spl : ndarray, optional
        Root mean square of Sound Pressure Level
    
    Returns:
    --------
    stats_df : pandas.DataFrame
        DataFrame containing statistics for each variable
    """
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
    
    return stats_df


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
    p_ref = 1e-6  # Reference pressure for underwater (1 µPa in Pascals)
    psd_db = 10 * np.log10(psd / (p_ref**2))
    
    return f, psd_db


def compute_spectral_metrics(audio_data, fs=None, window='hann', window_length=1.0, 
                             overlap=0.5, scaling='density', low_f=None, high_f=None):
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
    low_f : float, optional
        Lower frequency bound for bandpass filtering in Hz
    high_f : float, optional
        Upper frequency bound for bandpass filtering in Hz
    
    Returns:
    --------
    tuple
        (f, t, Sxx_db, rms_psd_f, percentiles, spl, rms_spl_t, bb_spl)
        - f: frequencies array
        - t: time array
        - Sxx_db: spectrogram in dB re 1 µPa²/Hz
        - rms_psd_f: RMS levels for each frequency bin in dB
        - percentiles: dictionary of percentile levels (1%, 5%, 50%, 95%, 99%) for each frequency bin
        - spl: Sound Pressure Level over time and frequency
        - rms_spl_t: RMS SPL over time
        - bb_spl: Broadband Sound Pressure Level over time
    """
    # Handle input data
    if hasattr(audio_data, 'samples'):
        sample_rate = audio_data.sample_rate
        # Apply bandpass filter if frequency bounds are provided
        if low_f is not None and high_f is not None:
            filtered_audio = audio_data.bandpass(low_f=low_f, high_f=high_f, order=4)
            samples = filtered_audio.samples
        else:
            samples = audio_data.samples
    else:
        samples = audio_data
        if fs is None:
            raise ValueError("Sample rate (fs) must be provided when input is not an Audio object")
        sample_rate = fs
        # Note: Bandpass filtering not applied for numpy arrays as it requires the audio_object interface
    
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
    
    # Small epsilon to avoid log of zero
    epsilon = np.finfo(float).eps
    
    # Compute SPL
    spl = 10 * np.log10(psd / (p_ref**2) + epsilon)

    # Compute RMS SPL over frequencies (for each time bin)
    rms_spl_t = 10 * np.log10(np.mean(10**(spl / 10), axis=0))
    
    # Convert spectrogram to dB re 1 µPa²/Hz
    psd_db = 10 * np.log10(psd / (p_ref**2) + epsilon)
    
    # Calculate RMS levels for each frequency bin (averaging over time)
    rms_psd_f = 10 * np.log10(np.mean(psd, axis=1) / (p_ref**2) + epsilon)
    
    # Calculate percentiles for each frequency bin
    percentiles = {
        "1%": np.percentile(psd_db, 1, axis=1),
        "5%": np.percentile(psd_db, 5, axis=1),
        "50%": np.percentile(psd_db, 50, axis=1),
        "95%": np.percentile(psd_db, 95, axis=1),
        "99%": np.percentile(psd_db, 99, axis=1)
    }
    
    # Broadband Level
    # Frequency resolution
    delta_f = np.diff(f)[0] if len(f) > 1 else f[0]  # Assuming uniform frequency spacing
    
    # Integrate PSD across all frequency bins
    power_per_time = np.sum(psd * delta_f, axis=0)  # Sum over frequency axis

    # Convert to SPL in dB
    bb_spl = 10 * np.log10(power_per_time / (p_ref**2) + epsilon)
    
    return f, t, psd_db, rms_psd_f, percentiles, spl, rms_spl_t, bb_spl


def compute_toctave_by_band(audio_data, low_f=None, high_f=None,
                           window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Calculate 1/3 octave band power analysis over time.

    Parameters:
    -----------
    audio_data : object
        Audio data object with attributes:
        - samples: Pressure data as ndarray
        - sample_rate: Sampling frequency in Hz
    low_f : float or None, optional
        Lower frequency bound in Hz. Default is 20 Hz if None.
    high_f : float or None, optional
        Upper frequency bound in Hz. Default is Nyquist frequency if None.
    window : str, optional
        Window type ('hann', 'hamming', 'blackman', etc.), default='hann'
    window_length : float, optional
        Length of window in seconds, default=1.0
    overlap : float, optional
        Overlap fraction between windows (0 to 1), default=0.5
    scaling : str, optional
        Scaling type ('density' or 'spectrum'), default='density'

    Returns:
    --------
    center_f : ndarray
        Center frequencies of the bands in Hz
    t : ndarray
        Time vector in seconds
    psd_db : ndarray
        Power spectral density matrix in dB (time x frequencies)
    rms_psd_f : ndarray
        RMS power spectral density in dB for each frequency band
    """
    # Extract data and sample rate from audio_data object
    samples = audio_data.samples
    fs = audio_data.sample_rate

    # Ensure samples is 1D array
    if len(samples.shape) > 1:
        if samples.shape[1] > samples.shape[0]:
            samples = samples.T
        samples = samples[:, 0]

    # Set default frequency range if not provided
    if low_f is None:
        low_f = 20.0  # Default lower bound (20 Hz is typical for audio)
    if high_f is None:
        high_f = fs / 2  # Nyquist frequency

    # Generate 1/3 octave band center frequencies and band edges
    def generate_third_octave_bands(low_f, high_f):
        """
        Generate center frequencies and band edges for 1/3 octave bands
        based on the given low_f and high_f.
        """
        fraction = 3  # Fixed for 1/3 octave bands
        factor = 2 ** (1 / fraction)  # Bandwidth factor

        # Initialize list to store center frequencies
        center_f = []

        # Start with the lowest frequency
        f_current = low_f

        # Generate center frequencies until we exceed high_f
        while f_current <= high_f:
            center_f.append(f_current)
            f_current *= factor

        # Convert to numpy array
        center_f = np.array(center_f)

        # Calculate band edges
        band_edges = np.zeros((len(center_f), 2))
        for i, fc in enumerate(center_f):
            band_edges[i, 0] = fc * (2 ** (-1 / (2 * fraction)))  # Lower edge
            band_edges[i, 1] = fc * (2 ** (1 / (2 * fraction)))   # Upper edge

        return center_f, band_edges

    # Generate 1/3 octave bands
    center_f, band_edges = generate_third_octave_bands(low_f, high_f)

    # Calculate window parameters
    samples_per_window = int(window_length * fs)
    hop_samples = int(samples_per_window * (1 - overlap))

    # Ensure window length is valid
    if samples_per_window > len(samples):
        samples_per_window = len(samples)
        hop_samples = samples_per_window // 2

    # Create window function
    if window == 'hann':
        win = signal.windows.hann(samples_per_window)
    elif window == 'hamming':
        win = signal.windows.hamming(samples_per_window)
    elif window == 'blackman':
        win = signal.windows.blackman(samples_per_window)
    else:
        win = signal.get_window(window, samples_per_window)

    # Calculate number of frames
    num_frames = 1 + (len(samples) - samples_per_window) // hop_samples

    # Initialize output arrays
    num_bands = len(center_f)
    psd = np.zeros((num_frames, num_bands))
    t = np.arange(num_frames) * hop_samples / fs

    # Calculate FFT frequencies
    fft_freqs = np.fft.rfftfreq(samples_per_window, d=1/fs)

    # Process each frame
    for i in range(num_frames):
        start = i * hop_samples
        end = start + samples_per_window

        if end > len(samples):
            break

        # Apply window
        frame = samples[start:end] * win

        # Compute FFT
        fft_data = np.fft.rfft(frame)

        # Compute power spectrum
        if scaling == 'density':
            # Power spectral density (V^2/Hz)
            power = np.abs(fft_data)**2 / (fs * np.sum(win**2))
        else:  # 'spectrum'
            # Power spectrum (V^2)
            power = np.abs(fft_data)**2 / np.sum(win**2)

        # Calculate power in each frequency band
        for j, (fc, edges) in enumerate(zip(center_f, band_edges)):
            # Find frequency indices within this band
            band_low, band_high = edges
            indices = (fft_freqs >= band_low) & (fft_freqs <= band_high)

            # Sum power in band
            if np.any(indices):
                psd[i, j] = np.sum(power[indices])

    # Reference pressure for underwater acoustics
    p_ref = 1e-6  # 1 µPa in Pa
    
    # Convert PSD to dB
    epsilon = np.finfo(float).eps  # Small constant to avoid log10(0)
    psd_db = 10 * np.log10(psd / (p_ref**2) + epsilon)

    # Calculate RMS PSD for each frequency band and convert to dB
    rms_psd = np.sqrt(np.mean(psd**2, axis=0))
    rms_psd_f = 10 * np.log10(rms_psd / (p_ref**2) + epsilon)

    return center_f, t, psd_db, rms_psd_f


def compute_toctave_by_nominal_freq(audio_data, center_f, window='hann', window_length=1.0, overlap=0.5, scaling='density'):
    """
    Apply a 1/3-octave bandpass filter based on the center frequency, then calculate
    the power spectral density (PSD) and related metrics.

    Parameters:
    -----------
    audio_data : object
        Audio data object with attributes:
        - samples: Audio samples as ndarray
        - sample_rate: Sampling frequency in Hz
    center_f : float
        The center frequency for the octave band (e.g., 63 Hz)
    window : str, optional
        Window type ('hann', 'hamming', etc.), default='hann'
    window_length : float, optional
        Length of window in seconds, default=1.0
    overlap : float, optional
        Overlap fraction between windows (0 to 1), default=0.5
    scaling : str, optional
        Scaling type ('density' or 'spectrum'), default='density'

    Returns:
    --------
    f : ndarray
        Frequencies of the spectrogram
    t : ndarray
        Time values of the spectrogram
    psd_db : ndarray
        Power spectral density in dB re 1 µPa²/Hz
    rms_psd_f : ndarray
        RMS of PSD across time for each frequency
    spl : ndarray
        Sound Pressure Level over time and frequency
    rms_spl_t : ndarray
        RMS SPL over time
    bb_spl : ndarray
        Broadband Sound Pressure Level over time
    """
    # Calculate the lower and upper frequency limits for the 1/3-octave band
    f_low = center_f * 2**(-1/6)
    f_high = center_f * 2**(1/6)

    # Apply the bandpass filter to the audio data
    filtered_audio = audio_data.bandpass(low_f=f_low, high_f=f_high, order=4)
    samples = filtered_audio.samples
    fs = filtered_audio.sample_rate

    # Calculate FFT parameters
    nperseg = int(fs * window_length)
    noverlap = int(nperseg * overlap)

    # Calculate spectrogram
    f, t, psd = signal.spectrogram(
        samples, 
        fs,
        window=window, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling=scaling
    )
    
    # Reference pressure for underwater acoustics
    p_ref = 1e-6  # 1 µPa in Pa
    
    # Small constant to avoid log10(0)
    epsilon = np.finfo(float).eps
    
    # Compute SPL
    spl = 10 * np.log10(psd / (p_ref**2) + epsilon)

    # Compute RMS SPL over frequencies (for each time bin)
    rms_spl_t = 10 * np.log10(np.mean(10**(spl / 10), axis=0))
    
    # Convert spectrogram to dB re 1 µPa²/Hz
    psd_db = 10 * np.log10(psd / (p_ref**2) + epsilon)
    
    # Calculate RMS levels for each frequency bin (averaging over time)
    rms_psd_f = 10 * np.log10(np.mean(psd, axis=1) / (p_ref**2) + epsilon)
    
    # Broadband Level
    # Frequency resolution
    delta_f = np.diff(f)[0] if len(f) > 1 else f[0]  # Assuming uniform frequency spacing

    # Integrate PSD across all frequency bins
    power_per_time = np.sum(psd * delta_f, axis=0)  # Sum over frequency axis

    # Convert to SPL in dB
    bb_spl = 10 * np.log10(power_per_time / (p_ref**2) + epsilon)
    
    return f, t, psd_db, rms_psd_f, spl, rms_spl_t, bb_spl