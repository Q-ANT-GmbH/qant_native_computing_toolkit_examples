# Function Learning with Fourier Layers
This project demonstrates function learning using a Fourier neural network executed on the Q.ANT Native Computing Toolkit.

We here use the Q.ANT Native Computing Toolkit to implement PyTorch layers that allow training on CPU / GPU and straightforward evaluation on the NPU.
A basic understanding of PyTorch and Fourier series is recommended.

## Why Fourier Layers Excel on NPU Hardware

Q.ANT's NPU evaluates nonlinear functions such as cosine directly in the optical domain using photonic components — no iterative numerical approximation needed. This makes Fourier layers a natural fit for the hardware.


## Overview
Notebook: function_learning_on_npu.ipynb \
Contains the complete workflow for training and evaluating a Fourier neural network.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash 
pip install -r requirements.txt 
```

## Usage
Run the notebook:
Open function_learning_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.
