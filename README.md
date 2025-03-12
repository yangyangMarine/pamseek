# pamseek

`pamseek` is an underwater Passive Acoustic Monitoring (PAM) data analysis repo designed to facilitate reproducible reporting, writing, and analysis of sound metrics from audio files. This package aims to streamline the analysis of underwater soundscape data, enabling more efficient data processing and interpretation.

## Features

- **Batch Loading Audio Files**: Efficiently load and process multiple audio files for analysis.
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

