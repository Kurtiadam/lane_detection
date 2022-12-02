import numpy as np
import cv2 as cv
from constants import PreprocConst


class Preproc:
    def __init__(self, preproc_consts: PreprocConst = PreprocConst):
        self.DOWNSCALE_TARGET_RES = np.array(
            [640, 480])  # width x height in pixels
        self.CROP_VERT_START = preproc_consts.CROP_VERT_START.value
        self.CROP_HOR_START = preproc_consts.CROP_HOR_START.value

    def downscale(self, frame: np.ndarray) -> np.ndarray:
        """Function to downscale the frame.

        Args:
            frame (np.ndarray): Frame to downscale

        Returns:
            downscaled (np.ndarray): Downscaled frame
        """
        downscaled = cv.resize(frame, self.DOWNSCALE_TARGET_RES)
        return downscaled

    def extract_roi(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Function to extract region of interest from the frame.

        Args:
            frame (np.ndarray): Frame to change

        Returns:
            cropped (np.ndarray): ROI of frame
            discarded (np.ndarray): Cut frame portion for later reconstruction
        """
        # .shape[0] = height, .shape[1] = width 
        cropped = frame[self.CROP_VERT_START:(
            frame.shape[0]), self.CROP_HOR_START:(frame.shape[1]-self.CROP_HOR_START)]
        discarded = frame[0:self.CROP_VERT_START, 0:(frame.shape[1])]
        return cropped, discarded

    def extract_lanes(self, cropped: np.ndarray,
        lower_yellow = np.array([0,100,100]),
        upper_yellow = np.array([100,200,255]),
        lower_white = np.array([0,122,25]),
        upper_white = np.array([101,255,150])) -> np.ndarray: # OpenCV halves the Hue values to fit 0-180
        """Function to convert the input image from BGR to HLS color space and then mask out everything other than the white/yellow lanes.

        Args:
            cropped (np.ndarray): Input frame to transform
            lower_yellow (np.ndarray, optional): Lower threshold limit for the yellow color. Defaults to np.array([25,70,100]).
            upper_yellow (np.ndarray, optional): Upper threshold limit for the yellow color. Defaults to np.array([70,200,255]).
            lower_white (np.ndarray, optional): Lower threshold limit for the white color. Defaults to np.array([27,122,25]).
            upper_white (np.ndarray, optional): Upper threshold limit for the white color. Defaults to np.array([101,255,150]).

        Returns:
            combined (np.ndarray): Extracted lanes lines.
            hls_yellow (np.ndarray): Extracted yellow lanes.
        """
        hls = cv.cvtColor(cropped, cv.COLOR_BGR2HLS)
        mask_yellow = cv.inRange(hls, lower_yellow, upper_yellow)
        mask_white = cv.inRange(hls, lower_white, upper_white)
        hls_yellow = cv.bitwise_and(cropped,cropped, mask = mask_yellow)
        hls_white = cv.bitwise_and(cropped,cropped, mask = mask_white)
        combined = cv.bitwise_or(hls_white,hls_yellow)
        return combined, hls_yellow
