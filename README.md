# Lane-detection system using Python and OpenCV

## Overview
This repository contains lane detection system made as a home project. The detection is implemented using traditional (non-learning) computer vision techniques and with the usage of Python and mainly the OpenCV library.

The project follows the following steps to detect lanes:
* **Preprocessing**
    1. Downscaling
    2. Cropping
    3. Extracting region of interest
    4. Transformation into HLS colorspace, tresholding
    5. Transformation into birdseye view
    6. Morphological opening
* **Feature extraction**
    1. Histogram creation
    2. Lane searching with sliding windows
    3. Polynomial line fitting
    4. Lane indicating and instuction giving to the driver

## Dependencies

* Python 3.10 or higher
* Numpy
* OpenCV
* Matplotlib
* Scipy

You can install these dependencies using the included requirements.txt file. (*$ pip install -r requirements.txt*)

## How to run
Run the main.py file