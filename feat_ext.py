import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class FeatExtract:
    def make_histogram(self, opened: np.ndarray) -> np.ndarray:
        """This function makes a histogram out of the inputted frame on which the user can see the vertical pixel count of each column of pixels.

        Args:
            opened (np.ndarray): Input frame.

        Returns:
            histogram (np.ndarray): Resulting histogram.
        """
        histogram = np.sum(opened[opened.shape[0]//2:,:], axis = 0) #Histogram of the bottom half
        midpoint = np.int32(histogram.shape[0]/2)
        leftxbase = np.argmax(histogram[:midpoint])
        rightxbase = np.argmax(histogram[midpoint:]) + midpoint
        return histogram
