# hdc2021_LGD
Our submission for the HDC2021. 

# Install 

Install the package using:

```
pip install -e .
```
Make sure to have git-lfs installed to pull the weight files for the model.

# Usage 

Prediction on images in a folder can be done using:

```
python deblurrer/main.py path-to-input-files path-to-output-files step
```


# Method

Learned Gradient Descent using a Bokeh Blur as a forward model.

# Examples


# Requirements 

* numpy = 1.20.3
* pytorch = 1.9.0 
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0

# Authors