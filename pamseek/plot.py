import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import defaultdict
from matplotlib.ticker import FuncFormatter


def plot_psd(f, pxx_db, xscale='log', yscale='linear', width=8, height=4,
             title='Power Spectral Density', grid=True, xlim=None, ylim=None, 
             save=False, filename='psd_plot.png', dpi=300, color='b'):
    """
    Plots the Power Spectral Density (PSD) analysis result.
    
    Parameters:
    -----------
    f : numpy.ndarray
        Frequency array
    pxx_db : numpy.ndarray
        PSD values in dB re 1 µPa²/Hz
    ... [rest of docstring remains unchanged]
    """
    fig = plt.figure(figsize=(width, height))
    plt.plot(f, pxx_db, color=color)
    
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB re 1 µPa²/Hz]')
    plt.grid(grid)
    plt.tight_layout()
    
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return None

def plot_spectrogram(f, t, sxx_db, xscale='linear', yscale='log', width=8, height=4,
                     title='Spectrogram', grid=True, xlim=None, ylim=(1, 48000), 
                     cmap='viridis', vmin=None, vmax=None, colorbar_label='PSD [dB re 1 µPa²/Hz]',
                     save=False, filename='spectrogram_plot.png', dpi=300,
                     downsample=True, segment_length=60):
    """
    Plots a spectrogram from time-frequency analysis with optional downsampling.
    """
    if downsample and len(t) > 1:
        dt = t[1] - t[0]
        bins_per_segment = max(1, int(segment_length / dt))
        num_segments = max(1, int(np.ceil(len(t) / bins_per_segment)))
        
        sxx_db_downsampled = np.zeros((sxx_db.shape[0], num_segments))
        t_downsampled = np.zeros(num_segments)
        
        for i in range(num_segments):
            start_idx = i * bins_per_segment
            end_idx = min((i + 1) * bins_per_segment, sxx_db.shape[1])
            sxx_db_downsampled[:, i] = np.mean(sxx_db[:, start_idx:end_idx], axis=1)
            t_downsampled[i] = np.mean(t[start_idx:end_idx]) if start_idx < len(t) else t[-1]
        
        t_plot, sxx_db_plot = t_downsampled, sxx_db_downsampled
    else:
        t_plot, sxx_db_plot = t, sxx_db
    
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

def plot_spectrum_line(x, y, percentiles=None, xscale='log', yscale='linear', 
                       width=8, height=4, title='Spectrum Line Plot', 
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
    
    xlabel = xlabel if xlabel is not None else ('Frequency [Hz]' if percentiles else 'Time [s]')
    ylabel = ylabel if ylabel is not None else ('PSD [dB re 1 µPa²/Hz]' if percentiles else 'SPL [dB]')
    
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

def plot_box(x, y, time_unit=1, time_unit_label='seconds', width=8, height=4,
             title='Broadband SPL Box Plot', grid=True, xlim=None, ylim=None, 
             save=False, filename='bb_spl_box_plot.png', dpi=300, 
             xlabel=None, ylabel=None, showfliers=True, color='skyblue', max_bins=20):
    """
    """
    x_converted = x / time_unit
    xlabel = xlabel if xlabel is not None else f'Time [{time_unit_label}]'
    ylabel = ylabel if ylabel is not None else 'Broadband SPL [dB re 1 µPa]'
    
    time_min, time_max = np.min(x_converted), np.max(x_converted)
    time_range = time_max - time_min
    bin_size = max(1, time_range / max_bins)
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    grouped_data = defaultdict(list)
    for t, spl in zip(x_converted, y):
        bin_index = int((t - time_min) / bin_size)
        bin_center = time_min + (bin_index + 0.5) * bin_size
        grouped_data[bin_center].append(spl)
    
    bin_centers = sorted(grouped_data.keys())
    data_to_plot = [grouped_data[center] for center in bin_centers]
    box_width = bin_size * 0.8
    
    bp = ax.boxplot(data_to_plot, positions=bin_centers, patch_artist=True, 
                    showfliers=showfliers, showmeans=False, showcaps=True, widths=box_width)
    
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=0.8)
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        margin = time_range * 0.05
        ax.set_xlim(time_min - margin, time_max + margin)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    def format_time(val, pos):
        seconds = val * time_unit
        if time_unit >= 86400:
            return f"{seconds/86400:.1f}d"
        elif time_unit >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"
        elif time_unit >= 60:
            minutes = int(seconds // 60)
            seconds_part = int(seconds % 60)
            return f"{minutes}m{seconds_part:02d}s"
        else:
            return f"{int(seconds)}s"
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(10, len(bin_centers))))
    if len(bin_centers) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)
    plt.tight_layout()
    
    if save:
        full_path = os.path.abspath(filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {full_path}")
    
    plt.show()
    return None