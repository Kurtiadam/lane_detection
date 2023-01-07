import cv2 as cv
import numpy as np
import os


class FrameDB:
    """Class for interaction with the video file"""

    def __init__(self, path: str = 'example_material\\example_video.mp4') -> None:
        """Capturing the video

        Args:
            path (str): Location of the example video
        """
        self.streaming = True
        self.cap = cv.VideoCapture(path)
        self.mod_frame_array = []

    def import_frame(self) -> tuple[bool, np.ndarray]:
        """Function to read and show the individual frames

        Returns:
            self.streaming (bool): Flag for indicating ongoing video rendering
            base (np.ndarray): Individual frames
        """
        ret, base = self.cap.read()
        self.streaming = self.cap.isOpened()
        return self.streaming, base

    def reconsturct_store(self, frame):
        height, width, layers = frame.shape
        frame_upscaled = cv.resize(frame, (width*4, height*4))
        self.mod_frame_array.append(frame_upscaled)
        return self.mod_frame_array

    def save_to_database(self, frame_array):
        height, width, layers = frame_array[0].shape
        size = (width, height)
        os.mkdir('output')
        os.chdir('output')
        out = cv.VideoWriter('detected_lanes.avi',
                             cv.VideoWriter_fourcc(*'XVID'), 24.0, size)
        for frame in frame_array:
            out.write(frame)
        out.release()
