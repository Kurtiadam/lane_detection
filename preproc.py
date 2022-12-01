import numpy as np
import cv2 as cv
from constants import ResModConst


class ResMod:
    def __init__(self, res_mod_consts: ResModConst = ResModConst):
        self.DOWNSCALE_TARGET_RES = np.array(
            [640, 480])  # width x height in pixels
        self.CROP_VERT_START = res_mod_consts.CROP_VERT_START.value
        self.CROP_HOR_START = res_mod_consts.CROP_HOR_START.value

    def downscale(self, frame: np.ndarray) -> np.ndarray:
        """Function to downscale the frame

        Args:
            frame (np.ndarray): Frame to downscale

        Returns:
            downscaled (np.ndarray): Downscaled frame
        """
        downscaled = cv.resize(frame, self.DOWNSCALE_TARGET_RES)
        return downscaled

    def extract_roi(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function to extract region of interest from the frame

        Args:
            frame (np.ndarray): Frame to change

        Returns:
            cropped (np.ndarray): ROI of frame
            discarded (np.ndarray): Cut frame portion for later reconstruction
        """
        # .shape[0] = height, .shape[1] = width 
        cropped = frame[self.CROP_VERT_START:(
            frame.shape[0]), self.CROP_HOR_START:(frame.shape[1])]
        discarded = frame[0:self.CROP_VERT_START, 0:(frame.shape[1])]
        return cropped, discarded
