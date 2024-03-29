import cv2 as cv
import numpy as np
import os
from os import path


class FrameDB:
    """Class for interaction with video files."""

    def __init__(self, import_path: str = 'example_material\\example_video.mp4') -> None:
        """Initializing contants and video capture.

        Args:
            import_path (str): Location of the example video.
        """
        self.streaming = True
        self.cap = cv.VideoCapture(import_path)
        # Array for storing individual frames to later save it in a video file
        self.mod_frame_array = []

    def import_frame(self) -> tuple[bool, np.ndarray]:
        """Reads and shows the individual frames.

        Returns:
            self.streaming (bool): Flag for indicating ongoing video rendering. True means ongoing rendering.
            base (np.ndarray): Imported frame.
        """
        ret, base = self.cap.read()
        self.streaming = self.cap.isOpened()
        return self.streaming, base

    def reconsturct_store(self, frame: np.ndarray) -> list[np.ndarray]:
        """Rescales and saves individual frames for later saving of the output file.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            self.mod_frame_array (list[np.ndarray]): Array of reconstructed frames to save.
        """
        height, width = frame.shape[:2]
        frame_upscaled = cv.resize(frame, (width*4, height*4))
        self.mod_frame_array.append(frame_upscaled)
        return self.mod_frame_array

    @staticmethod
    def save_to_database(frame_array: list[np.ndarray]) -> None:
        """Saves the given array of frames into a video into a new folder named 'output'.

        Args:
            frame_array (list[np.ndarray]): Array of input frames.
        """
        height, width = frame_array[0].shape[:2]
        size = (width, height)
        # If an 'output' folder doesn't exist creates it and saves the video there (detected_lanes.avi)
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
