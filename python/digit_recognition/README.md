# Digit Recognition
This project demonstrates digit recognition using a two-layer fully-connected neural network, executed on an NPU.

## Overview
Notebook: digit_recognition_on_npu.ipynb
Contains the complete workflow for evaluating the digit recognition model.

Dataset: MNIST
The model classifies input images of handwritten digits (0–9) from the MNIST dataset.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt 
```

## Usage
Run the notebook:
Open digit_recognition_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_digit_recog.py contains automated tests for digit recognition using PyTest.

```bash
pytest test_digit_recog.py 
```

# Description
A trained two-layer fully-connected neural network is used to classify a input image into one of ten classes (digit no.).
The used data set is MNIST (https://en.wikipedia.org/wiki/MNIST_database).

