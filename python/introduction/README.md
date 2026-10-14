# Introduction to the Q.ANT Native Computing Toolkit Python API
This project introduces the core operations of the Q.ANT Native Computing Toolkit Python API, validating each of them against a PyTorch reference implementation.

## Overview
Notebook: intro.ipynb
Walks through the SDK's mathematical and AI/neural-network operations - element-wise multiplication, scaled sinusoidal and KAN layers, linear layers, 2D convolution and transposed convolution, bias, activation functions (ReLU, Sigmoid, Softmax), pooling, and batch normalization - comparing each against PyTorch in bfloat16 precision.

Notebook: intro_memory_management.ipynb
Walks through the different levels of memory management.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt 
```

## Usage
Run the notebook:
Open intro.ipynb/intro_memory_management.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_nb.py executes the notebook and asserts that every operation's error against the PyTorch reference stays below its expected threshold.

```bash
pytest test_nb.py 
```

# Description
Each section of the notebook creates random test data, computes results using both the Q.ANT SDK and PyTorch, compares the outputs, and records error metrics used by the automated test.
