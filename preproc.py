import numpy as np
import cv2 as cv


class Preproc:
    def downscale(self, frame: np.ndarray, DOWNSCALE_TARGET_RES=np.array([640, 480])) -> np.ndarray:
        """_Function to downscale the frame.
        Args:
            frame (np.ndarray): Frame to downscale.
            DOWNSCALE_TARGET_RES (np.array, optional): Downscale target resolution. Defaults to np.array([640, 480]).

        Returns:
            downscaled (np.ndarray): Downscaled frame
        """
        downscaled = cv.resize(frame, DOWNSCALE_TARGET_RES)
        return downscaled

    def extract_roi(self, frame: np.ndarray, CROP_VERT_START=250, CROP_HOR_START=25) -> tuple[np.ndarray, np.ndarray]:
        """Function to extract region of interest from the frame.
        Args:
            frame (np.ndarray): Frame to change
            CROP_VERT_START (int, optional): Vertical starting point of the cropping. Defaults to 250.
            CROP_HOR_START (int, optional): Horizontal starting point of the cropping. Defaults to 25.

        Returns:
            cropped (np.ndarray): ROI of frame
            discarded (np.ndarray): Cut frame portion for later reconstruction
        """
        # .shape[0] = height, .shape[1] = width
        cropped = frame[CROP_VERT_START:(
            frame.shape[0]), CROP_HOR_START:(frame.shape[1]-CROP_HOR_START)]
        discarded = frame[0:CROP_VERT_START, 0:(frame.shape[1])]
        return cropped, discarded

    def extract_lanes(self, cropped: np.ndarray,
                      lower_yellow=np.array([0, 100, 100]),
                      upper_yellow=np.array([100, 200, 255]),
                      lower_white=np.array([0, 122, 25]),
                      upper_white=np.array([101, 255, 150])) -> np.ndarray:  # OpenCV halves the Hue values to fit 0-180
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
        hls_yellow = cv.bitwise_and(cropped, cropped, mask=mask_yellow)
        hls_white = cv.bitwise_and(cropped, cropped, mask=mask_white)
        combined = cv.bitwise_or(hls_white, hls_yellow)
        return combined, hls_yellow

    def birdseye_transform(self, input, leftupper=[150, 100], rightupper=[350, 100], leftlower=[50, 180], rightlower=[400, 180]) -> tuple[np.ndarray, int, list[list[int]]]:
        """Method to transform the POV frame to a birdseye view.

        Args:
            input (np.ndarray): Input frame to transform.
            leftupper (list, optional): Left upper corner point for the transformation algorithm. Defaults to [150,100].
            rightupper (list, optional): Right upper corner point for the transformation algorithm. Defaults to [350,100].
            leftlower (list, optional): Left lower corner point for the transformation algorithm. Defaults to [50,180].
            rightlower (list, optional): Right lower corner point for the transformation algorithm. Defaults to [400,180].

        Returns:
            birdseye (np.ndarray): Birdseyeview transformed frame.
            Minv (int): Inverse matrix for backtransformation into POV.
            birdview_points (list[list[int]]): Selected points for birdeye view transformation.
        """
        height, width = input.shape[:2]
        birdview_points = [leftupper, rightupper, leftlower, rightlower]

        src = np.float32([leftupper, leftlower, rightupper, rightlower])
        dst = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        Matrix = cv.getPerspectiveTransform(src, dst)
        Minv = cv.getPerspectiveTransform(dst, src)
        birdseye = cv.warpPerspective(input, Matrix, (width, height))
        return birdseye, Minv, birdview_points  # Felülnézeti kép, inverz mátrix

    def make_binary(self, combined: np.ndarray, thresh: int = 80, kernel: np.ndarray = np.ones((5, 5))) -> tuple[np.ndarray, np.ndarray]:
        """Grayscale image creation, blurring and thresholding method.

        Args:
            combined (np.ndarray): Input frame
            thresh (int, optional): Threshold level limit [0-255]. Defaults to 80.

        Returns:
            blurred (np.ndarray): Blurred frame.
            binary (np.ndarray): Binary result frame (blurred + tresholded).
        """
        frame_gray = cv.cvtColor(combined, cv.COLOR_BGR2GRAY)
        # <-SLOW | FASTER ->cv2.GaussianBlur(frame_gray,(7,7),cv2.BORDER_DEFAULT)
        blurred = cv.bilateralFilter(frame_gray, 9, 120, 120)
        binary = cv.threshold(blurred, thresh, 255, cv.THRESH_BINARY)[1]
        # canny = cv2.Canny(blurred,90,150) #100,200 basic tresholds <- SLOW | FASTER -> Sobel operator
        #canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
        return blurred, binary

        # Opening (erosion + dilation)
    def opening(self, birdseye: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.uint8), iterations: int = 1) -> np.ndarray:
        """Morphologic opening: first erosion then dilation.

        Args:
            birdseye (np.ndarray): Frame to erode.
            kernel (np.ndarray, optional): Eroding kernel. Defaults to np.ones((3,3),np.uint8).
            iterations (int, optional): Number of erosion iterations. Defaults to 1.

        Returns:
            opened (np.ndarray): Eroded frame.
        """
        eroded = cv.erode(birdseye, kernel, iterations)
        opened = cv.dilate(eroded, kernel, iterations)
        return opened
