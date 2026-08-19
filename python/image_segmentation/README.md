# Image Segmentation
This project demonstrates image segmentation using a UNet neural network, executed on an NPU.

## Overview
Notebook: image_segmentation_on_npu.ipynb
Contains the complete workflow for evaluating the image segmentation model.

Dataset: Brain MRI Segmentation
The model classifies the pixels of the input image into to classes.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash 
pip install -r requirements.txt 
```

## Usage
Run the notebook:
Open image_segmentation_on_npu.ipynb in Jupyter Notebook or JupyterLab to explore the code and results interactively.

## Testing
PyTest test case:
The file test_image_segementation.py contains automated tests for image segmentation using PyTest.

```bash 
pytest test_image_segementation.py 
```

# Description
A trained UNet neural network is used to detect abnormalities in brain MRI recordings.
The used data set is brain MRI (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

