# Image Recognition
This project demonstrates image recognition using a ResNet-18 neural network, executed on an NPU.

## Overview
Notebook: image_recognition_on_npu.ipynb
Contains the complete workflow for evaluating the image recognition model.

Dataset: ImageNet
The model classifies input pictures and classifies the input into on of the 1000 classes of ImageNet.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash 
pip install -r requirements.txt 
```

## Usage
Run the notebook:
Open image_recognition_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_image_recog.py contains automated tests for image recognition using PyTest.

```bash 
pytest test_image_recog.py 
```

# Description
A trained ResNet-18 neural network is used to classify a input image into one of thousand classes.
The used data set is ImageNet (https://image-net.org/).

