# Signal Denoising with the DFT
This project demonstrates how to denoise a noisy sine wave in the frequency domain, using the discrete Fourier transform (DFT) of the Q.ANT Native Computing Toolkit for both the forward and the inverse transform.

## Overview
Notebook: signal_denoising_on_npu.ipynb \
Generates a sine wave with strong Gaussian noise, transforms it to the frequency domain with `q_native.dft`, removes all frequencies above a cutoff with a simple low-pass mask and transforms the result back to the time domain. The time and frequency domain are plotted at each step.

The NPU only provides a forward DFT. The inverse DFT is computed with the same operation using complex conjugation:

$$\text{IDFT}(X) = \frac{1}{N} \, \overline{\text{DFT}\left(\overline{X}\right)}$$

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Run the notebook:
Open signal_denoising_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_signal_denoising.py executes the notebook and asserts that the mean squared error between the denoised signal and the clean sine is below 0.1 and less than 10% of the error of the noisy input.

```bash
pytest test_signal_denoising.py
```
