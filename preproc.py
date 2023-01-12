import numpy as np
import cv2 as cv


class Preproc:
    def __init__(self, mode: int = 1) -> None:
        """Initialization of the preprocessing algorithm.

        Args:
            mode (int, optional): mode: 0: lane detection based on color filtered frames; 1: lane detection based on the lightness channel. Defaults to 1.
        """
        self.mode = mode
    @staticmethod
    def downscale(frame: np.ndarray, DOWNSCALE_TARGET_RES=np.array([640, 480])) -> np.ndarray:
        """Downscales a frame.
        Args:
            frame (np.ndarray): Frame to downscale.
            DOWNSCALE_TARGET_RES (np.array, optional): Downscale target resolution. Defaults to np.array([640, 480]) ([width,height]).

        Returns:
            downscaled (np.ndarray): Downscaled frame.
        """
        downscaled = cv.resize(frame, DOWNSCALE_TARGET_RES)
        return downscaled

    @staticmethod
    def extract_roi(frame: np.ndarray, CROP_VERT_START: int=250, CROP_HOR_START:int=0) -> tuple[np.ndarray, np.ndarray]:
        """Extracts the region of interest from the frame.
        Args:
            frame (np.ndarray): Input frame.
            CROP_VERT_START (int, optional): Vertical starting point of the cropping [pixel]. Defaults to 250.
            CROP_HOR_START (int, optional): Horizontal starting point of the cropping [pixel]. Defaults to 0.

        Returns:
            cropped (np.ndarray): Region of interest of frame.
            discarded (np.ndarray): Cut frame portion for later reconstruction.
        """
        cropped = frame[CROP_VERT_START:(
            frame.shape[0]), CROP_HOR_START:(frame.shape[1]-CROP_HOR_START)]
        discarded = frame[0:CROP_VERT_START,
                          CROP_HOR_START:(frame.shape[1]-CROP_HOR_START)]
        return cropped, discarded

    def colorspace_transform(self, input_frame: np.ndarray,
                             LOWER_YELLOW=np.array([15, 80, 100]),
                             UPPER_YELLOW=np.array([30, 200, 255]),
                             LOWER_WHITE=np.array([0, 120, 0]),
                             UPPER_WHITE=np.array([180, 255, 255])) -> np.ndarray:
        """Converts the input image from BGR to HLS color space and then masks out the pixels outside the allowed range.

        Args:
            input_frame (np.ndarray): Input frame to transform.
            LOWER_YELLOW (np.ndarray, optional): Lower threshold limit for the yellow color. Defaults to np.array([15, 80, 100]).
            UPPER_YELLOW (np.ndarray, optional): Upper threshold limit for the yellow color. Defaults to np.array([30, 200, 255]).
            LOWER_WHITE (np.ndarray, optional): Lower threshold limit for the white color. Defaults to np.array([0, 120, 0]).
            UPPER_WHITE (np.ndarray, optional): Upper threshold limit for the white color. Defaults to np.array([180, 255, 255]).

        Returns:
            color_filtered (np.ndarray): The HLS frame with kept whites or only the L channel of it.
            hls_yellow (np.ndarray): HLS frame with kept yellow colors.
        """
        hls = cv.cvtColor(input_frame, cv.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        mask_yellow = cv.inRange(hls, LOWER_YELLOW, UPPER_YELLOW)
        mask_white = cv.inRange(hls, LOWER_WHITE, UPPER_WHITE)
        hls_yellow = cv.bitwise_and(input_frame, input_frame, mask=mask_yellow)
        hls_white = cv.bitwise_and(input_frame, input_frame, mask=mask_white)
        color_filtered = cv.bitwise_or(hls_white, hls_yellow)

        # Changing between operation modes, see README or the main function's comments
        if self.mode == 1:
            color_filtered = l_channel
        return color_filtered, hls_yellow

    @staticmethod
    def birdseye_transform(input_frame:np.ndarray, PERS_TRANS_LEFTUPPER:np.array, PERS_TRANS_RIGHTUPPER:np.array, PERS_TRANS_LEFTLOWER:np.array, PERS_TRANS_RIGHTLOWER:np.array) -> tuple[np.ndarray, np.ndarray, np.array]:
        """Transforms the POV frame to a birdseye view.

        Args:
            input_frame (np.ndarray): Input frame to transform.
            PERS_TRANS_LEFTUPPER (np.array): Left upper corner point for the transformation algorithm.
            PERS_TRANS_RIGHTUPPER (np.array): Right upper corner point for the transformation algorithm.
            PERS_TRANS_LEFTLOWER (np.array): Left lower corner point for the transformation algorithm.
            PERS_TRANS_RIGHTLOWER (np.array): Right lower corner point for the transformation algorithm.

        Returns:
            birdseye (np.ndarray): Birdseyeview transformed frame.
            Minv (np.ndarray): Inverse matrix for backtransformation into POV.
            birdview_points (np.array): Selected points for birdeye view transformation.
        """
        height, width = input_frame.shape[:2]
        birdview_points = [PERS_TRANS_LEFTUPPER, PERS_TRANS_RIGHTUPPER, PERS_TRANS_LEFTLOWER, PERS_TRANS_RIGHTLOWER]

        src = np.float32([PERS_TRANS_LEFTUPPER, PERS_TRANS_LEFTLOWER, PERS_TRANS_RIGHTUPPER, PERS_TRANS_RIGHTLOWER])
        dst = np.float32(
            [[0, 0], [150, height], [width, 0], [width-150, height]])
        Matrix = cv.getPerspectiveTransform(src, dst)
        Minv = cv.getPerspectiveTransform(dst, src)
        birdseye = cv.warpPerspective(input_frame, Matrix, (width, height))
        return birdseye, Minv, birdview_points

    def make_binary(self, input_frame: np.ndarray, SOBEL_THRESH_LOW: int = 80) -> tuple[np.ndarray, np.ndarray]:
        """Creates the binary output frame with extracted edges. 
        Includes a grayscale image transformation if needed (input frame channel > 1), Gaussian filtering, Sobel edge detection and thresholding.

        Args:
            input_frame (np.ndarray): Input frame.
            SOBEL_THRESH_LOW (int, optional): Threshold level limit [0-255]. Defaults to 80.

        Returns:
            binary (np.ndarray): Binary resulting frame.
        """

        try:
            input_frame = cv.cvtColor(input_frame, cv.COLOR_BGR2GRAY)
        except:
            pass

        # blurred_bilateral = cv.bilateralFilter(input_frame, 5, 50, 50)
        blurred = cv.GaussianBlur(input_frame, (5, 5), cv.BORDER_DEFAULT)

        if self.mode == 1:
            SOBEL_THRESH_LOW = 40
        sobel_x = cv.Sobel(blurred, cv.CV_16S, 1, 0, ksize=3,
                           scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_sobel_x = cv.convertScaleAbs(sobel_x)
        
        binary = np.zeros_like(abs_sobel_x)
        binary[(abs_sobel_x >= SOBEL_THRESH_LOW) & (abs_sobel_x <= 255)] = 255
        return binary

    @staticmethod
    def opening(input_frame: np.ndarray, EROSION_KERNEL: np.ndarray = np.ones((3, 3), np.uint8), DILATION_KERNEL: np.ndarray = np.ones((3, 3), np.uint8), iterations: int = 1) -> np.ndarray:
        """Morphologic opening: first erosion then dilation.

        Args:
            input_frame (np.ndarray): Input frame to morphologicly open.
            EROSION_KERNEL (np.ndarray, optional): Kernel for erosion, must be same odd numbers: (1,1);(3,3) etc. Defaults to np.ones((3, 3), np.uint8).
            DILATION_KERNEL (np.ndarray, optional): Kernel for dilation, must be same odd numbers: (1,1);(3,3) etc. Defaults to np.ones((3, 3), np.uint8).
            iterations (int, optional): Number of opening iterations. Defaults to 1.

        Returns:
            np.ndarray: Opened output frame.
        """
        eroded = cv.erode(input_frame, EROSION_KERNEL, iterations)
        opened = cv.dilate(eroded, DILATION_KERNEL, iterations)
        return opened
