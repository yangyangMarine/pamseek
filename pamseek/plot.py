import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_psd(f, Pxx_dB, xscale='log', yscale='linear', width=12, height=6,
             title='Power Spectral Density', grid=True, xlim=None, ylim=None, 
             save=False, filename='psd_plot.png', dpi=300, color='b'):
    """
    Plots the Power Spectral Density (PSD) analysis result.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frequency array
    Pxx_dB : numpy.ndarray
        PSD values in dB re 1 µPa²/Hz
    xscale : str, optional
        Scale for x-axis ('log' or 'linear', default: 'log')
    yscale : str, optional
        Scale for y-axis ('log' or 'linear', default: 'linear')
    width : int, optional
        Figure width in inches (default: 12)
    height : int, optional
        Figure height in inches (default: 6)
    title : str, optional
        Plot title (default: 'Power Spectral Density')
    grid : bool, optional
        Whether to show grid (default: True)
    xlim : tuple, optional
        Limits for x-axis (min, max) (default: None)
    ylim : tuple, optional
        Limits for y-axis (min, max) (default: None)
    save : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saved plot (default: 'psd_plot.png')
    dpi : int, optional
        Resolution of saved plot in dots per inch (default: 300)
    color : str, optional
        Line color for the plot (default: 'b')
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig = plt.figure(figsize=(width, height))
    
    # Plot the PSD
    plt.plot(f, Pxx_dB, color=color)
    
    # Set x and y axis scales
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [dB re 1 µPa²/Hz]')
    plt.grid(grid)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {os.path.abspath(filename)}")
    
    plt.show()
    return fig

def plot_spectrogram(f, t, Sxx_dB, xscale='linear', yscale='log', width=12, height=6,
                    title='Spectrogram', grid=True, xlim=None, ylim=[1, 12000], 
                    cmap='viridis', vmin=None, vmax=None, colorbar_label='PSD [dB re 1 µPa²/Hz]',
                    save=False, filename='spectrogram_plot.png', dpi=300):
    """
    Plots a spectrogram from time-frequency analysis.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frequency array
    t : numpy.ndarray
        Time array
    Sxx_dB : numpy.ndarray
        Spectrogram values in dB re 1 µPa²/Hz
    xscale : str, optional
        Scale for x-axis ('log' or 'linear', default: 'linear')
    yscale : str, optional
        Scale for y-axis ('log' or 'linear', default: 'log')
    width : int, optional
        Figure width in inches (default: 12)
    height : int, optional
        Figure height in inches (default: 6)
    title : str, optional
        Plot title (default: 'Spectrogram')
    grid : bool, optional
        Whether to show grid (default: True)
    xlim : tuple, optional
        Limits for x-axis (min, max) (default: None)
    ylim : tuple, optional
        Limits for y-axis (min, max) (default: [1, 12000])
    cmap : str, optional
        Colormap for spectrogram (default: 'viridis')
    vmin : float, optional
        Minimum value for color scaling (default: None)
    vmax : float, optional
        Maximum value for color scaling (default: None)
    colorbar_label : str, optional
        Label for colorbar (default: 'PSD [dB re 1 µPa²/Hz]')
    save : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saved plot (default: 'spectrogram_plot.png')
    dpi : int, optional
        Resolution of saved plot in dots per inch (default: 300)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig = plt.figure(figsize=(width, height))
    
    # Plot the spectrogram
    pcm = plt.pcolormesh(t, f, Sxx_dB, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set x and y axis scales
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    # Add colorbar
    cbar = plt.colorbar(pcm)
    cbar.set_label(colorbar_label)
    
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.grid(grid)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {os.path.abspath(filename)}")
    
    plt.show()
    return fig

def plot_psd_with_percentiles(f, Pxx_dB, percentiles=None, xscale='log', yscale='linear', 
                             width=12, height=6, title='PSD with Percentiles', grid=True, 
                             xlim=None, ylim=None, save=False, filename='psd_percentiles.png', 
                             dpi=300, colors=None):
    """
    Plots the Power Spectral Density (PSD) along with percentile statistics.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frequency array
    Pxx_dB : numpy.ndarray
        PSD values in dB re 1 µPa²/Hz
    percentiles : dict, optional
        Dictionary of percentile values (e.g., {'50%': values, '95%': values})
    xscale : str, optional
        Scale for x-axis ('log' or 'linear', default: 'log')
    yscale : str, optional
        Scale for y-axis ('log' or 'linear', default: 'linear')
    width : int, optional
        Figure width in inches (default: 12)
    height : int, optional
        Figure height in inches (default: 6)
    title : str, optional
        Plot title (default: 'PSD with Percentiles')
    grid : bool, optional
        Whether to show grid (default: True)
    xlim : tuple, optional
        Limits for x-axis (min, max) (default: None)
    ylim : tuple, optional
        Limits for y-axis (min, max) (default: None)
    save : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saved plot (default: 'psd_percentiles.png')
    dpi : int, optional
        Resolution of saved plot in dots per inch (default: 300)
    colors : dict, optional
        Dictionary mapping percentile labels to colors
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig = plt.figure(figsize=(width, height))
    
    # Default colors if not provided
    if colors is None:
        colors = {
            "1%": "lightblue",
            "5%": "skyblue",
            "50%": "blue",
            "95%": "darkblue",
            "99%": "navy"
        }
    
    # Plot main PSD if provided
    if Pxx_dB is not None:
        plt.plot(f, Pxx_dB, 'k-', label='Mean PSD', linewidth=2)
    
    # Plot percentiles if provided
    if percentiles is not None:
        for label, values in percentiles.items():
            color = colors.get(label, 'gray')
            plt.plot(f, values, '-', label=f'{label} Percentile', color=color, alpha=0.7)
    
    # Set x and y axis scales
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density [dB re 1 µPa²/Hz]')
    plt.grid(grid)
    plt.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {os.path.abspath(filename)}")
    
    plt.show()
    return fig

def analyze_and_plot_audio(audio_data, fs=None, analysis_type='both', window='hann', 
                          window_length=1.0, overlap=0.5, plot_kwargs=None, save=False, 
                          psd_filename='psd_plot.png', spec_filename='spectrogram_plot.png'):
    """
    Performs acoustic analysis and plots results in one function call.
    
    Parameters:
    -----------
    audio_data : opensoundscape.Audio or numpy.ndarray
        Audio object or 1-dimensional array of audio samples in Pascals
    fs : float, optional
        Sampling frequency in Hz. Required if audio_data is numpy array.
    analysis_type : str, optional
        Type of analysis to perform ('psd', 'spectrogram', or 'both', default: 'both')
    window : str or tuple, optional
        Window function to use (default: 'hann')
    window_length : float, optional
        Length of each window in seconds (default: 1.0)
    overlap : float, optional
        Overlap factor between 0 and 1 (default: 0.5 for 50% overlap)
    plot_kwargs : dict, optional
        Additional keyword arguments for plotting functions
    save : bool, optional
        Whether to save plots (default: False)
    psd_filename : str, optional
        Filename for PSD plot (default: 'psd_plot.png')
    spec_filename : str, optional
        Filename for spectrogram plot (default: 'spectrogram_plot.png')
        
    Returns:
    --------
    dict
        Dictionary containing analysis results and plot figures
    """
    from scipy import signal
    
    # Handle input data
    if hasattr(audio_data, 'samples'):
        samples = audio_data.samples
        sample_rate = audio_data.sample_rate
    else:
        samples = audio_data
        if fs is None:
            raise ValueError("Sample rate (fs) must be provided when input is not an Audio object")
        sample_rate = fs
    
    # Default plot kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    
    results = {}
    
    # Calculate segment parameters
    nperseg = int(sample_rate * window_length)
    noverlap = int(nperseg * overlap)
    
    # Perform PSD analysis if requested
    if analysis_type in ['psd', 'both']:
        # Calculate PSD using Welch's method
        f, Pxx = signal.welch(
            samples, 
            fs=sample_rate,
            window=window, 
            nperseg=nperseg,
            noverlap=noverlap, 
            scaling='density'
        )
        
        # Convert PSD from Pa²/Hz to dB re 1 µPa²/Hz
        P_ref = 1e-6  # Reference pressure (1 µPa in Pascals)
        Pxx_db = 10 * np.log10(Pxx / (P_ref**2) + np.finfo(float).eps)
        
        # Store results
        results['psd'] = {
            'f': f,
            'Pxx_db': Pxx_db
        }
        
        # Plot PSD
        psd_kwargs = {k: v for k, v in plot_kwargs.items() if k not in ['cmap', 'vmin', 'vmax', 'colorbar_label']}
        fig_psd = plot_psd(f, Pxx_db, save=save, filename=psd_filename, **psd_kwargs)
        results['psd_fig'] = fig_psd
    
    # Perform spectrogram analysis if requested
    if analysis_type in ['spectrogram', 'both']:
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(
            samples, 
            fs=sample_rate,
            window=window, 
            nperseg=nperseg,
            noverlap=noverlap, 
            scaling='density'
        )
        
        # Convert spectrogram to dB re 1 µPa²/Hz
        P_ref = 1e-6  # Reference pressure (1 µPa in Pascals)
        epsilon = np.finfo(float).eps
        Sxx_db = 10 * np.log10(Sxx / (P_ref**2) + epsilon)
        
        # Calculate statistics
        rms_level = 10 * np.log10(np.mean(Sxx, axis=1) / (P_ref**2) + epsilon)
        
        percentiles = {
            "1%": np.percentile(Sxx_db, 1, axis=1),
            "5%": np.percentile(Sxx_db, 5, axis=1),
            "50%": np.percentile(Sxx_db, 50, axis=1),
            "95%": np.percentile(Sxx_db, 95, axis=1),
            "99%": np.percentile(Sxx_db, 99, axis=1)
        }
        
        # Store results
        results['spectrogram'] = {
            'f': f,
            't': t,
            'Sxx_db': Sxx_db,
            'rms_level': rms_level,
            'percentiles': percentiles
        }
        
        # Plot spectrogram
        spec_kwargs = {k: v for k, v in plot_kwargs.items() if k != 'color'}
        fig_spec = plot_spectrogram(f, t, Sxx_db, save=save, filename=spec_filename, **spec_kwargs)
        results['spectrogram_fig'] = fig_spec
        
        # Plot PSD with percentiles
        if analysis_type == 'spectrogram':  # Only do this if not already plotting PSD
            fig_perc = plot_psd_with_percentiles(f, rms_level, percentiles, 
                                               save=save, 
                                               filename='psd_percentiles.png', 
                                               **{k: v for k, v in plot_kwargs.items() if k not in ['cmap', 'vmin', 'vmax', 'colorbar_label']})
            results['percentiles_fig'] = fig_perc
    
    return results