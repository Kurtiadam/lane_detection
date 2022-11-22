import cv2 as cv
import numpy as np

class Visualizer():
    @staticmethod
    def render(frame: np.ndarray):
        """
        This function shows the image in a new window to the user.
        """
        cv.imshow('Lane detection test video', frame)