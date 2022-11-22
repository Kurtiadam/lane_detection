import cv2 as cv
import numpy as np


class FrameDB:
    """Class for interaction with the video file"""

    def __init__(self, path: str) -> None:
        """Capturing the video

        Args:
            path (str): Location of the example video
        """
        self.streaming = True
        self.cap = cv.VideoCapture(path)

    def import_frame(self) -> tuple[bool, np.ndarray]:
        """Function to read and show the individual frames

        Returns:
            self.streaming (bool): Flag for indicating ongoing video rendering
            base (np.ndarray): Individual frames
        """
        ret, base = self.cap.read()
        self.streaming = self.cap.isOpened()
        return self.streaming, base

    # def save_to_database():
