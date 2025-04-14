import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import defaultdict
from matplotlib.ticker import FuncFormatter


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

def plot_spectrogram(ds, xscale='linear', yscale='log', width=8, height=4, title='Spectrogram', 
                     grid=True, xlim=None, ylim=(1, 48000), cmap='viridis', vmin=None, vmax=None, 
                     colorbar_label='PSD [dB re 1 µPa²/Hz]', save=False, filename='spectrogram_plot.png', 
                     dpi=300, segment_length=60):
    """
    Plots a spectrogram from xarray Dataset with time averaging.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing 'psd_db' DataArray with dimensions ['time', 'frequency']
    xscale : str, optional
        Scale of the x-axis ('linear' or 'log')
    yscale : str, optional
        Scale of the y-axis ('linear' or 'log')
    width, height : float, optional
        Figure width and height in inches
    title : str, optional
        Plot title
    grid : bool, optional
        Whether to display grid lines
    xlim, ylim : tuple or None, optional
        Limits for x and y axes
    cmap : str, optional
        Colormap name
    vmin, vmax : float or None, optional
        Minimum and maximum values for colormap
    colorbar_label : str, optional
        Label for the colorbar
    save : bool, optional
        Whether to save the plot to a file
    filename : str, optional
        Filename for the saved plot
    dpi : int, optional
        DPI for the saved plot
    segment_length : int, optional
        Number of time points to average together
    """
    # Extract data from the dataset
    f = ds.frequency.values
    t = ds.time.values
    sxx_db = ds.psd_db.values
    
    # Time averaging
    if segment_length > 1 and len(t) > 1:
        num_segments = max(1, int(np.ceil(len(t) / segment_length)))
        sxx_db_averaged = np.zeros((sxx_db.shape[0], num_segments))
        t_averaged = np.zeros(num_segments)
        
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, sxx_db.shape[1])
            sxx_db_averaged[:, i] = np.mean(sxx_db[:, start_idx:end_idx], axis=1)
            t_averaged[i] = np.mean(t[start_idx:end_idx]) if start_idx < len(t) else t[-1]
        
        t_plot, sxx_db_plot = t_averaged, sxx_db_averaged
    else:
        t_plot, sxx_db_plot = t, sxx_db
    
    # Create the plot
    fig = plt.figure(figsize=(width, height))
    pcm = plt.pcolormesh(t_plot, f, sxx_db_plot, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.colorbar(pcm, label=colorbar_label)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.grid(grid)
    plt.tight_layout()
    
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return None