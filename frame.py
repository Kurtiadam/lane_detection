import cv2 as cv
import numpy as np
import os
from os import path


class FrameDB:
    """Class for interaction with the video file"""

    def __init__(self, import_path: str = 'example_material\\example_video.mp4') -> None:
        """Capturing the video

        Args:
            import_path (str): Location of the example video
        """
        self.streaming = True
        self.cap = cv.VideoCapture(import_path)
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
        height, width = frame.shape[:2]
        frame_upscaled = cv.resize(frame, (width*4, height*4))
        self.mod_frame_array.append(frame_upscaled)
        return self.mod_frame_array

    def save_to_database(self, frame_array):
        height, width = frame_array[0].shape[:2]
        size = (width, height)
        if not path.exists('output'):
            os.mkdir('output')
        os.chdir('output')
        print("Please wait until the output is saved!")
        out = cv.VideoWriter('detected_lanes.avi',
                             cv.VideoWriter_fourcc(*'XVID'), 24.0, size)
        for frame in frame_array:
            out.write(frame)
        print("The video was successfully saved!")
        out.release()
