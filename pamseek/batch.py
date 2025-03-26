import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import warnings
import scipy.signal as signal
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytz

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
        Bit depth of the audio signal (default: 24)
        
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

def process_audio_files(path, sensitivity, gain, fs=None, 
                         window='hann', window_length=1.0, overlap=0.5, 
                         scaling='density', low_f=None, high_f=None, 
                         output_dir=None):
    """
    Process multiple WAV files in a directory to compute Power Spectral Density (PSD).
    
    Parameters:
    -----------
    path : str
        Directory containing WAV files
    sensitivity : float
        Hydrophone sensitivity in dB re 1V/µPa
    gain : float
        Additional gain applied to the signal
    fs : float, optional
        Sampling frequency (if None, uses audio file's native sample rate)
    window : str, optional
        Window function for PSD computation (default: 'hann')
    window_length : float, optional
        Length of window in seconds (default: 1.0)
    overlap : float, optional
        Overlap between windows (default: 0.5)
    scaling : str, optional
        PSD scaling method (default: 'density')
    low_f : float, optional
        Low-frequency bandpass filter limit
    high_f : float, optional
        High-frequency bandpass filter limit
    output_dir : str, optional
        Directory to save output files (default: input path)
    
    Returns:
    --------
    xarray.Dataset
        Combined dataset with PSD and broadband SPL information
    """
    # Suppress numpy runtime warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Store original working directory and change to input path
    original_dir = os.getcwd()
    os.chdir(path)
    
    # Set output directory
    output_dir = output_dir or path
    os.makedirs(output_dir, exist_ok=True)

    # Find WAV files
    wav_files = sorted(glob.glob('*.wav'))
    if not wav_files:
        os.chdir(original_dir)
        raise ValueError(f"No .wav files found in {path}")

    # Reference pressure and epsilon for calculations
    P_REF = 1e-6  # reference pressure
    EPSILON = np.finfo(float).eps

    # Containers for combined results
    combined_f = []
    combined_t = []
    combined_psd_db = []
    combined_bb_spl_db = []

    # Process each WAV file
    for i, single_file in enumerate(wav_files, 1):
        # Progress tracking
        progress = int((i/len(wav_files))*100)
        file_count = f"({i}/{len(wav_files)})"
        progress_bar = "[" + "=" * (progress//2) + " " * (50 - progress//2) + "]"
        print(f"\rProcessing {file_count} {progress_bar} {progress}%", end="", flush=True)
        
        # Load and calibrate audio
        audio_object = Audio.from_file(single_file)
        timestamp_str = extract_timestamp_from_filename(single_file)
        start_time = pytz.timezone("UTC").localize(timestamp_str)

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
     
        # Save individual file result as NetCDF
        filename = os.path.splitext(single_file)[0] + '_processed.nc'
        output_path = os.path.join(output_dir, filename)
        
        ds = xr.Dataset(
            {
                'psd_db': (['frequency'], psd_db),
                'bb_spl_db': bb_spl_db,
                'time': start_time.timestamp()
            },
            coords={'frequency': f},
            attrs={
                'sensitivity': sensitivity,
                'gain': gain,
                'sample_rate': sample_rate,
                'window': window,
                'window_length': window_length,
                'overlap': overlap,
                'scaling': scaling,
                'low_f': low_f if low_f is not None else 'None',
                'high_f': high_f if high_f is not None else 'None'
            }
        )
        ds.to_netcdf(output_path)

    # Print completion message
    print(f"\n\nProcessing complete! {len(wav_files)} files processed.\n")

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
            'high_f': high_f if high_f is not None else 'None'
        }
    )

    os.chdir(original_dir)
    
    return combined_ds


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

def plot_psd(x, y, percentiles=None, xscale='log', yscale='linear', 
                       width=8, height=4, title='PSD', 
                       grid=True, xlim=None, ylim=None, save=False, filename='spectrum_line_plot.png', 
                       dpi=300, colors=None, xlabel=None, ylabel=None):
    """
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
    
    fig = plt.figure(figsize=(width, height))
    plt.plot(x, y, label='RMS Level', color='red', linestyle='--', linewidth=2)
    
    if percentiles:
        for label, values in percentiles.items():
            color = colors.get(label, 'gray')
            plt.plot(x, values, '-', label=f'{label} Percentile', color=color, alpha=0.7)
    
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.legend(loc='best')
    plt.tight_layout()
    
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return None


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
    """
    # Frequency resolution
    df = float(ds.frequency[1] - ds.frequency[0])  # Hz
    
    # Convert PSD from dB to linear units
    psd_db = ds.psd_db
    
    psd_linear = 10 ** (psd_db / 10)  # Convert to μPa²/Hz
    
    # Integrate over frequency and convert to dB
    p_total = psd_linear.sum(dim='frequency') * df
    bb_spl = 10 * np.log10(p_total)  # dB re 1 μPa

    time = ds.time
    # Return as xarray with time coordinate
    return xr.Dataset(
        data_vars={
            'bb_spl': (('time'), bb_spl.data),
        },
        coords={'time': ds.time}
    )


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

def plot_bb_spl(times, rms_spl, percentiles=None, 
                width=8, height=4, 
                title='Broadband SPL Segmentation', 
                grid=True, ylim=None, 
                save=False, 
                filename='broadband_spl_plot.png', 
                dpi=300):
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
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Define percentile colors
    percentile_colors = {
        "1%": "lightblue",
        "5%": "skyblue",
        "50%": "blue",
        "95%": "darkblue",
        "99%": "navy"
    }
    
    # Plot RMS Level
    ax.plot(times, rms_spl, label='RMS Level', color='red', linestyle='--', linewidth=2)
    
    # Plot percentiles if provided
    if percentiles is not None:
        percentile_labels = ["1%", "5%", "50%", "95%", "99%"]
        for i, label in enumerate(percentile_labels):
            ax.plot(times, percentiles[:, i], 
                   '-', 
                   label=f'{label} Percentile', 
                   color=percentile_colors[label], 
                   alpha=0.7)
    
    # Format x-axis as time
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Broadband SPL (dB re 1 µPa)')
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid and legend
    ax.grid(grid, linestyle=':', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return fig


def boxplot_bb_spl(ds, segment_duration='30min', width=8, height=4,
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
    ax.set_xticklabels([edge.strftime('%Y-%m-%d %H:%M') for edge in bin_edges], 
                       rotation=45, ha='right')

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

def compute_toctave_band(audio_data, center_f):
    """
    Apply a 1/3-octave bandpass filter based on the center frequency

    Parameters:
    -----------
    audio_data : object
    center_f : float
        The center frequency for the octave band (e.g., 63 Hz)

    Returns:
    --------
    
    """
    # Calculate the lower and upper frequency limits for the 1/3-octave band
    f_low = center_f * 2**(-1/6)
    f_high = center_f * 2**(1/6)

    # Apply the Butterworth bandpass filter to the audio data
    audio_data = audio_data.bandpass(low_f=f_low, high_f=f_high, order=4)
 
    return audio_data