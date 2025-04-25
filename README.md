# pamseek

`pamseek` is an underwater Passive Acoustic Monitoring (PAM) data analysis package designed to process large datasets using parallel processing. It streamlines and standardizes the processing, analysis, and reporting of underwater soundscape metrics in a fast and reproducible way.

**This package leverages Python's multiprocessing module to efficiently handle large audio datasets by utilizing multiple CPU cores.** This parallel processing capability significantly speeds up PAM data analysis, making it ideal for high-throughput workflows and long-term monitoring projects.

## Features

- **Parallel Audio Processing**: Efficiently load and process multiple audio files simultaneously using multicore processing
- **Spectrogram Visualization**: Generate high-quality time-frequency representations of acoustic data
- **LTSA Analysis**: Calculate and visualize Long-Term Spectral Averages for extended monitoring periods
- **PSD Computation**: Perform Power Spectral Density analysis for detailed frequency domain assessment
- **SPL Measurement**: Calculate Sound Pressure Level metrics to quantify underwater sound intensity
- **Broadband and Band-Limited Analysis**: Conduct 1/3 octave band and other filter-based analyses
- **Signal Detection**: Integrate whistle detection functionality from [soundscape_IR](https://github.com/schonkopf/soundscape_IR.git)
- **Annotation Tools**: Process and modify annotations created with [Raven Software](https://www.ravensoundsoftware.com/)
- **Advanced Processing**: Implement hydrophone calibration, timestamp extraction, and various filtering techniques (Butterworth, Chebyshev, frequency domain)
- **Future Development**: Upcoming CNN classification capabilities based on [opensoundscape](https://github.com/kitzeslab/opensoundscape.git)

## Installation

Install `pamseek` using pip:

```bash
pip install pamseek
```

## Quick Start

```python
import pamseek

# Process audio files using multiple cores
DATA_PATH = r"E:\Hydrophone\2024-10"
ds_toctave_125Hz = pamseek.process_audio_files(
    path=DATA_PATH,
    sensitivity=-170.4,
    gain=2.05,
    fs=96000,
    window='hann',
    window_length=0.08533,
    overlap=0.5,
    scaling='density',
    low_f=111,
    high_f=140,
    n_processes=None,  # Uses all available cores by default
    output_dir="E:\Hydrophone\output",
    output_filename="OToctave_band_125Hz.nc"
)
```
Processing displays a progress bar and automatically skips corrupted files. Results are saved in NetCDF format with a confirmation message showing the file location.
![Speed Processing](hhttps://github.com/yangyangMarine/pamseek/blob/main/docs/batch.png)

## Examples

### LTSA (Long-Term Spectral Average)
```python
pamseek.plot_ltsa(ds)
```
![LTSA Example](https://github.com/yangyangMarine/pamseek/blob/main/docs/LTSA.png)

### SPL and 1/3 Octave Band Analysis
![Octave Band SPL](https://github.com/yangyangMarine/pamseek/blob/main/docs/OTband_SPL.png)

### Power Spectral Density (PSD)
![PSD Example](https://github.com/yangyangMarine/pamseek/blob/main/docs/PSD.png)

## Documentation

For detailed documentation and examples, visit our [documentation site](https://pamseek.readthedocs.io/) (coming soon).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

- [soundscape_IR](https://github.com/schonkopf/soundscape_IR.git) for whistle detection algorithms
- [opensoundscape](https://github.com/kitzeslab/opensoundscape.git) for annotation tools and CNN classification framework
- Contributors and researchers from the underwater acoustics community