import cv2 as cv

class Visualizer():
    @staticmethod
    def render(frame):
        """
        This function shows the image in a new window to the user.
        """
        cv.imshow('Lane detection test video', frame)