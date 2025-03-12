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
                     title='Spectrogram', grid=True, xlim=None, ylim=None, 
                     cmap='viridis', vmin=None, vmax=None, colorbar_label='PSD [dB re 1 µPa²/Hz]',
                     save=False, filename='spectrogram_plot.png', dpi=300,
                     downsample=False, segment_length=60):
    """
    Plots a spectrogram from time-frequency analysis with optional downsampling.

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
        Limits for y-axis (min, max). If None, defaults to [f.min(), f.max()].
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
    downsample : bool, optional
        Whether to downsample the data (default: False)
    segment_length : int, optional
        Length of each segment in seconds for downsampling (default: 60)

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if downsample:
        # Calculate the number of time bins per segment
        dt = t[1] - t[0]
        bins_per_segment = int(segment_length / dt)
        
        # Split the data into segments
        num_segments = int(np.ceil(Sxx_dB.shape[1] / bins_per_segment))
        Sxx_dB_downsampled = np.zeros((Sxx_dB.shape[0], num_segments))
        t_downsampled = np.zeros(num_segments)
        
        for i in range(num_segments):
            start_idx = i * bins_per_segment
            end_idx = min((i + 1) * bins_per_segment, Sxx_dB.shape[1])
            Sxx_dB_downsampled[:, i] = np.mean(Sxx_dB[:, start_idx:end_idx], axis=1)
            t_downsampled[i] = np.mean(t[start_idx:end_idx])
        
        # Use the downsampled data for plotting
        t_plot = t_downsampled
        Sxx_dB_plot = Sxx_dB_downsampled
    else:
        # Use the original data for plotting
        t_plot = t
        Sxx_dB_plot = Sxx_dB
    
    # Set y-axis limits if not provided
    if ylim is None:
        ylim = [f.min(), f.max()]  # Default to full frequency range

    fig = plt.figure(figsize=(width, height))
    
    # Plot the spectrogram
    pcm = plt.pcolormesh(t_plot, f, Sxx_dB_plot, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set x and y axis scales
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Set axis limits
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(ylim)  # Use the provided or default y-axis limits
    
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

def plot_psd_with_percentiles(f, Sxx_dB, percentiles=None, xscale='log', yscale='linear', 
                             width=12, height=6, title='PSD with Percentiles', grid=True, 
                             xlim=None, ylim=None, save=False, filename='psd_percentiles.png', 
                             dpi=300, colors=None):
    """
    Plots the Power Spectral Density (PSD) along with percentile statistics.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frequency array
    Sxx_dB : numpy.ndarray
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
                               
    # Calculate RMS of the PSD in dB
    rms_level = 10 * np.log10(np.mean(10**(Sxx_dB/10), axis=1))  # RMS for each frequency bin
                               
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
    
    # Plot main rms_level if provided
    if rms_level is not None:
        plt.plot(f, rms_level, label='RMS Level', color='red', linestyle='--', linewidth=2)
    
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
    plt.legend(loc='lower left')

    plt.tight_layout()
    
    if save:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {os.path.abspath(filename)}")
    
    plt.show()
    return fig
