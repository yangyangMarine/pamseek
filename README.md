# pamseek

`pamseek` is an underwater Passive Acoustic Monitoring (PAM) data analysis package designed to streamline and standardize the processing, analysis, and reporting of underwater soundscape metrics. 
**This package is built with multiprocessing support using Python’s multiprocessing module, allowing it to handle large audio datasets efficiently by leveraging multiple CPU cores.** This parallel processing capability significantly speeds up PAM data analysis, making it ideal for high-throughput workflows and long-term monitoring projects.

## Features

- **Batch Loading Audio Files**: Efficiently load and process multiple audio files for analysis, using multicore.
- **Spectrogram Plotting**: Visualize time-frequency representations of sound using spectrograms.
- **LTSA (Long-Term Spectral Average)**: Calculate and visualize the LTSA for sound data.
- **PSD (Power Spectral Density)**: Compute and analyze the PSD for frequency domain analysis.
- **SPL (Sound Pressure Level)**: Compute SPL to assess the intensity of underwater sounds.
- **Broadband Noise Level**: Perform noise level analysis, including 1/3 octave band analysis.
- **Whistle Detection**: Integrate whistle detection functionality from [soundscape_IR](https://github.com/schonkopf/soundscape_IR.git).
- **Raven Annotation Manipulation**: Modify annotations created with Raven Software ([Raven](https://www.ravensoundsoftware.com/)), developed by [opensoundscape](https://github.com/kitzeslab/opensoundscape.git).
- **CNN Classification**: Future integration of classification capabilities using Convolutional Neural Networks (CNN), developed by [opensoundscape](https://github.com/kitzeslab/opensoundscape.git).

## Installation

Install `pamseek` using pip:

```bash
pip install pamseek

import pamseek

# Multi core, speedy process.

DATA_PATH = r"E:\\Hydrophone\\2024-10"
ds_toctave_125Hz = process_audio_files(
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
    n_processes=None,
    output_dir="E:\\Hydrophone\\output",
    output_filename="OToctave_band_125Hz.nc"
)

# Load audio file
audio_data = pamseek.load_audio_files('path_to_audio_file.wav')

# Compute LTSA and SPL
ltsa = pamseek.compute_ltsa(audio_data)
spl = pamseek.compute_spl(audio_data)

# Plot spectrogram
pamseek.plot_spectrogram(audio_data)
```

## License


## Acknowledgments

TBC...
