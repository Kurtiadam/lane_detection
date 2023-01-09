import numpy as np
import cv2 as cv


class FeatExtract:
    @staticmethod
    def make_histogram(binary_input_frame: np.ndarray, HISTOGRAM_ROI_PROP: float = 1) -> np.ndarray:
        """Makes a histogram out of the inputted frame on which in every column the sum of positive pixels is shown.

        Args:
            HISTOGRAM_ROI_PROP (float, optional): Proportion of ROI to non-ROI of histograms vertical axis. (histogram height / HISTOGRAM_ROI_PROP)
            binary_input_frame (np.ndarray): Input frame.

        Returns:
            histogram (np.ndarray): Resulting histogram.
        """
        histogram = np.sum(binary_input_frame[round(binary_input_frame.shape[0]//HISTOGRAM_ROI_PROP):, :],
                           axis=0)
        return histogram

    @staticmethod
    def lane_search(opened: np.ndarray, histogram: np.ndarray, lane_type: str, NWINDOWS: int = 20, OFFSET: int = 15, MINPIX: int = 50, window_debug: bool = False) -> tuple[tuple[np.array, np.array], tuple[np.array, np.array], tuple[bool, bool], bool]:
        """Searching for lane with sliding windows.

        Args:
            opened (np.ndarray): Input frame. Has to be a binary image.
            histogram (np.ndarray): The binary input frames histogram.
            lane_type (str, optional): Color of the input frames lanes. ("combined" OR "yellow")
            NWINDOWS (int, optional): Number of sliding windows vertically for each lane (right and left). Defaults to 20.
            OFFSET (int, optional): Horizontal width of the rectangle from its middle point. Defaults to 15.
            MINPIX (int, optional): Minimum number of pixels in a sliding window to recognize as a lane piece. Defaults to 50.
            window_debug (bool, optional): Flag for the window debugging mode. (True = ON)

        Returns:
            left_points (list(np.array, np.array)): Found white pixels with the windows on the left side. (x coordinates, y coordinates)
            right_points (list(np.array, np.array)): Found white pixels with the windows on the right side. (x coordinates, y coordinates)
            sides_ok (list(bool,bool)): Indicator for each side if the number of adequate windows reaches the requirement. (True = adequate)
            yellow_lanes (bool): Flag for indicating yellow lanes on both sides. (True = yellow lanes on both side)
        """

        # Constants and variables for sliding window technique
        NWINDOWS = 20
        WINDOW_PROPORTION_TO_YELLOW = round(NWINDOWS * (1/3))
        WINDOW_PROPORTION_TO_POLY_FIT = round(NWINDOWS * (1/3))
        midpoint = np.int32(histogram.shape[0]/2)
        leftxbase = np.argmax(histogram[:midpoint])
        rightxbase = np.argmax(histogram[midpoint:]) + midpoint
        window_h = np.int32(opened.shape[0] / NWINDOWS)
        left_lane_inds = []
        right_lane_inds = []
        nwindow_ok_left = 0
        nwindow_ok_right = 0
        nwindow_yellow_left = 0
        nwindow_yellow_right = 0
        left_side_ok = False
        right_side_ok = False
        yellow_lanes = False

        # Get white pixels
        nonzero = opened.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current maximum number of white pixels
        leftx_current = leftxbase
        rightx_current = rightxbase

        # Looping through windows
        for window in range(NWINDOWS):
            # Lower and upper edge of the window
            win_y_low = opened.shape[0] - (window + 1) * window_h
            win_y_high = opened.shape[0] - window * window_h
            # Left and right edge coordinates of the windows
            win_xleft_left = leftx_current - OFFSET
            win_xleft_right = leftx_current + OFFSET
            win_xright_left = rightx_current - OFFSET
            win_xright_right = rightx_current + OFFSET

            # Drawing windows if window debugging is enabled
            if window_debug:
                cv.rectangle(opened, (win_xleft_left, win_y_low),
                             (win_xleft_right, win_y_high), (255, 255, 0), 1)
                cv.rectangle(opened, (win_xright_left, win_y_low),
                             (win_xright_right, win_y_high), (255, 255, 0), 1)

            # Picking the white pixels which the windows contain on both sides
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_left) & (nonzerox < win_xleft_right)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_left) & (nonzerox < win_xright_right)).nonzero()[0]

            # Making sure the number of contained pixels exceeds the minimum threshold
            # If it does, then get the x position of the lane with the mean values for the next window adjusting
            if len(good_left) > MINPIX:
                # Counting windows of the yellow frame with adequate number of pixels
                if lane_type == "yellow":
                    nwindow_yellow_left += 1
                # Changing the x base coordinate of the next window based on the mean of the current
                leftx_current = np.int32(np.mean(nonzerox[good_left]))
                # Counting windows with adequate number of pixels
                nwindow_ok_left += 1
            if len(good_right) > MINPIX:
                if lane_type == "yellow":
                    nwindow_yellow_right += 1
                rightx_current = np.int32(np.mean(nonzerox[good_right]))
                nwindow_ok_right += 1

            # Sum of good pixels
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

        # Sum of lane pixels in whole image on both sides
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Getting the x and y coordinates of lane pixels
        left_points = [nonzerox[left_lane_inds], nonzeroy[left_lane_inds]]
        right_points = [nonzerox[right_lane_inds], nonzeroy[right_lane_inds]]

        # Examination of adequate yellow windows count and assigning the yellow lane case
        if nwindow_yellow_left > WINDOW_PROPORTION_TO_YELLOW and nwindow_yellow_right > WINDOW_PROPORTION_TO_YELLOW:
            yellow_lanes = True

        # Signal if the number of adequate windows reaches the required limit
        if nwindow_ok_left > WINDOW_PROPORTION_TO_POLY_FIT:
            left_side_ok = True
        if nwindow_ok_right > WINDOW_PROPORTION_TO_POLY_FIT:
            right_side_ok = True
        sides_ok = [left_side_ok, right_side_ok]

        return left_points, right_points, sides_ok, yellow_lanes

    @staticmethod
    def poly_fit(left_points: tuple[np.array, np.array], right_points: tuple[np.array, np.array], left_fit_prev: np.ndarray, right_fit_prev: np.ndarray, nwindow_ok: bool) -> tuple[np.ndarray, np.ndarray]:
        """Fits a first degree polynomial on the given lane pixels.

        Args:
            left_points (tuple[np.array,np.array]): Lane pixels on the left side to fit polynomial on.
            right_points (tuple[np.array,np.array]): Lane pixels on the right side to fit polynomial on.
            left_fit_prev (np.ndarray): Polynomial fit of the lane pixels on the left side from the previous frame.
            right_fit_prev (np.ndarray): Polynomial fit of the lane pixels on the right side from the previous frame.
            nwindow_ok (bool): Flag for the reaching the required minimum windows to fit a polynomial on it. (left,right), (True = OK)

        Returns:
            left_fit (np.ndarray): Polynomial coefficient of the lane pixels on the left side.
            right_fit (np.ndarray): Polynomial coefficient of the lane pixels on the right side.
        """
        leftx, lefty = left_points[:]
        rightx, righty = right_points[:]
        left_side_ok, right_side_ok = nwindow_ok[:]

        # If adequate windows on both sides are reached, the poly fit will take place
        if left_side_ok and right_side_ok:
            left_fit = np.polyfit(lefty, leftx, 1)
            right_fit = np.polyfit(righty, rightx, 1)
        # If only one of the sides is adequate for poly fitting, the inadequate will get the previous frames poly fit
        elif left_side_ok and ~right_side_ok:
            left_fit = np.polyfit(lefty, leftx, 1)
            right_fit = right_fit_prev
        elif ~left_side_ok and right_side_ok:
            left_fit = left_fit_prev
            right_fit = np.polyfit(righty, rightx, 1)
        # If neither of the sides is adequate, the previous polynomial coefficient will be passed on
        else:
            left_fit = left_fit_prev
            right_fit = right_fit_prev
        return left_fit, right_fit

    @staticmethod
    def draw_lane_lines(input_frame: np.ndarray, birdseye: np.ndarray, Minv: np.ndarray, discarded: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray) -> tuple[int, np.ndarray]:
        """Draw the lane lines and the lane on the input frame.

        Args:
            input_frame (np.ndarray): Input frame to draw on.
            birdseye (np.ndarray): Perspective transformed frame to transform back into original POV view.
            Minv (np.ndarray): Inverse matrix for backtransformation.
            discarded (np.ndarray): Cut image portion that will be added to the input frame for reconstructing the original image.
            left_fit (np.ndarray): Polynomial coefficients of the left lane.
            right_fit (np.ndarray): _olynomial coefficients of the right lane.

        Returns:
            direction (int): Direction of the vehile from the lanes middle line. (0: in the middle, 1: vehicle is to the right side, -1: vehicle is to the left side)
            original (np.ndarray): Output frame with the indicated lanes transformed back into original POV. 
        """
        # Fitting the lines onto a plane
        ploty = np.linspace(0, input_frame.shape[0]-1, input_frame.shape[0])
        left_fitx = left_fit[0]*ploty + left_fit[1]
        right_fitx = right_fit[0]*ploty + right_fit[1]

        # Getting the lane points of both sides
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Avaraging the points to get the middle line of the lane
        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array(
            [np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        # Empty 3 channel frame to draw on
        warp_zero = np.zeros_like(birdseye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Marking the lane
        cv.fillPoly(color_warp, np.int_([pts]), (0, 120, 0))
        # Marking the middle line of the lane
        cv.fillPoly(color_warp, np.int_([pts_mean]), (255, 0, 0))

        # Transforming back the original image and adding the markings to it
        newwarp = cv.warpPerspective(
            color_warp, Minv, (input_frame.shape[1], input_frame.shape[0]))
        final = cv.addWeighted(input_frame, 1, newwarp, 0.5, 0)
        original = np.concatenate((discarded, final), axis=0)

        # Evaulating where the vehicle is compared to the middle line of the lane
        lookaway_distance = int(pts_mean.shape[1]/2)
        # shape: (1, 230, 2); second -1 index indicating the most farthest point the detector can show (lookaway_distance = -1)
        mpts = np.int32(pts_mean[-1][lookaway_distance][-2])
        deviation = original.shape[1] / 2 - abs(mpts)
        CENTER_OK = 50
        if abs(deviation) <= CENTER_OK:
            direction = 0
        elif deviation < -CENTER_OK:
            direction = -1
        else:
            direction = 1

        return direction, original
