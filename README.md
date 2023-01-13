# Lane-detection system using Python and OpenCV

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/211877265-3d5878a8-62fb-41af-8cf2-2613b83a06f2.png">
</p>

## Overview
This repository contains a lane detection system made as a home project. The detection is implemented using traditional (non-learning) computer vision techniques and with the usage of Python and mainly the OpenCV library.

## Dependencies

* Python 3.10 or higher
* Numpy
* OpenCV
* Matplotlib

You can install these dependencies using the included "requirements.txt" file. (*$ pip install -r requirements.txt*)

## How to run
Run the main.py file. 
At the main function definition in the source code you can toggle between different modes which affect the running of the program, these are:
* mode: 0: lane detection based on color filtered frames; 1: lane detection based on the lightness channel
* birdseye_view__points_debug: Visualizing the points which the perspective transform is based on
* histogram_debug: While enabled, by pressing the '*h*' key, the user can see the given binary frame's histogram
* sliding_windows_debug: Lets the user inspect the workings of the sliding windows technique
* save_result: saves the output video with the detected lanes into a new folder (Can take longer)
The running of the program can be stopped at any time by pressing the key '*q*'.

## Functioning of the algorithm
The project follows the following steps to detect lanes:
* **Preprocessing**
1. Downscaling
2. Cropping and extracting region of interest
3. Transformation into HLS colorspace, tresholding
4. Perspective transform
5. Blurring, Sobel filtering, binary image creation, morphological opening
* **Feature extraction**
6. Histogram creation
7. Lane searching with sliding windows
8. Polynomial line fitting
9. Lane indicating and instruction giving to the driver

### Downscaling
After the acqusition of the first frame from the input video file, it is downscaled into a lower resulution in order to achieve faster runtime performance, which can be crucial in embedded systems. The downscaling improves performance without a significant loss of information.

### Cropping and extracting region of interest

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212109803-f3feef85-3098-4b41-9b35-061d9b7291ee.png">
</p>

After the downscaling the frame will be cropped to remove unnecessary region of the image, only preserving the region of interest, i.e. the image portion which includes the road. The discarded portion will be save for later reconstruction of the original image with the lane detected on it.

### Transformation into HLS colorspace, tresholding

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212110239-495acc6e-d96a-4291-af3a-ea9a432decf5.png">
</p>

In order to separate lane colors from the rest of the image, the frame is converted from the RGB linear colorspace into a non-linear colorspace called HLS (Hue, Lightness, Saturation). Based on these channels the white and yellow regions of the image can be better isolated. For yellow lane detection the color yellow mainly with the hue and saturation channels will be isolated, for white lanes also the same process can be used. Another solution for white lane detection is to use only the lightness channel to filter out the very light pixels (near the whites). This doesn't work on yellow lanes as well as on whites, because of this for yellow lanes mainly the hue and saturation channels will be used.
The user can switch between the two modes for white lane detection by changing the mode variable in the main function: 0 means white color filtered frames, 1 means lightness channel based detection. Both of these techniques have their advantages:
White color filtering allows the concealment of shadows which can be distracting for the lane detection. While this helps in sunny circumstances, if the lane is in the shadow it won't be as likely to be detected as with the lightness channel detection. You can see this in the corresponding images (above L channel, below white (+yellow) color filtered images): On the lightness channel the lane piece covered by the shadow of the car in front is still visible, but the shadow also distracts the algorithm thinking that it is a straight line ergo the lane. This filtered out on the color filtered image, even tho the lane piece is not visible in the shadow.

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212061521-7737783a-1a15-41ae-a8c9-bd372888a51f.png">
</p>

### Perspective transform

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212156102-87dca140-3722-4b5a-84df-9b08c0cb7d00.png">
</p>

To achieve better results from the sliding window technique and the polynomial line fitting the POV image is transformed into a birds eye view like perspective. For this the coordinates of quadrangle verticles has to be given which will be transformed into a corresponding quadrangle vertices in the destination image.
The transformation matrix's inverse is saved for later reconstruction of the POV view. The user can enable the '*birdseye_view__points_debug*' flag with which one can see the selected region that will be transformed.

### Blurring, Sobel filtering, binary image creation, morphological opening
<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212156579-a0484b20-57c4-4a14-990a-d09eb2f6da04.png">
</p>

The resulting "birds eye view" image will be blurred with a Gaussian filter to remove white noise from the camera sensor. After this Sobel filtering with a vertical kernel will be carried out to detect the vertical lines in the image. The horizontal lines are not relevant since lanes are vertically displayed.
The resulting image is then thresholded into a binary image which only contains white and black pixels. As the preprocessings last step, the frame is morphologically opened (erosion + dilation) to further remove noise and enhance features.


### Histogram creation
<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212157001-31cba9b3-8314-43ff-a075-12f6d70e319c.png">
</p>

From the preprocessed binary frame a histogram will be created. This shows the white pixel count in every column of the image. Taking the highest spikes on both half sides of this histogram can serve as a likely starting point of a lane and also the horizontal position of the first sliding window.
The user can enable the '*histogram_debug*' flag to view a given frames histogram anytime by pressing the key '*h*' during the running of the program.

### Lane searching with sliding windows

<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212156849-c292ec5b-4c34-471b-82d7-bcbf9b8225e0.png">
</p>

A given number of windows will be created on both halves of the input image. The lowermost window's horizontal location will be based on the histograms peaks. White pixels inside these windows will be saved for fitting polynomials on them. Besides the lowermost windows the rest are positioned by taking the mean of white pixels' x coordinates, resulting in free horizontal sliding of the windows hence the name.
The adequate windows which contain enough white pixels to fit a line on them are counted and only after reaching a given amount will the polynomial fitting occur, if there is not enough adequate windows, the polynomial will be fitted on the last frame with adequate number of windows. The lane searching is done pararelly on the yellow lanes as well, if there are sufficient number of windows with positive pixels of the yellow lanes on *both* sides, the algorithm switches over yellow lane following, just like in real cases when the white lanes are corrected with yellow ones. Only one sided yellow lane doesn't result in this approach.
The user can enable the '*sliding_windows_debug*' to see the workings of the sliding window technique.

### Polynomial line fitting
A first degree polynomial line will be fitted on the resulting white pixels from the sliding windows. The number of adequate windows will be taken into account whether to fit on the current or the last frames white pixels.

### Lane indicating and instruction giving to the driver
<p align="center">
    <img src="https://user-images.githubusercontent.com/98428367/212157351-77bf7661-3e9d-4416-8e1c-d64efcb86add.png">
</p>

Lastly, the frame will be converted back into a POV view and on it will be displayed the detected lane with the middle line of it and also several messages and instructions for the user. On the left upper corner of the image the user receives messages whether the location of the vehicle in the lane is proper or what action the driver should make. If the algorithm follows yellow lanes, this will be indicated in the lower left corner. The FPS counter can be seen in the lower right corner.

## Discussion
This approach of lane detection is not usable reliably in a real traffic situation. There are multiple scenarios where this algorithm won't work e.g. very dark environment, lanes covered by other vehicles or weather factors like snow etc. Certainly better results can be achieved via learning solutions e.g. deep learning based approaches.

If any messages or proposals arise feel free to contact me or open a issue.
Thank you for your attention!
