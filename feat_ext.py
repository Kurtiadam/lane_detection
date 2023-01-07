import numpy as np
import cv2 as cv


class FeatExtract:
    def make_histogram(self, opened: np.ndarray) -> np.ndarray:
        """This function makes a histogram out of the inputted frame on which the user can see the vertical pixel count of each column of pixels.

        Args:
            opened (np.ndarray): Input frame.

        Returns:
            histogram (np.ndarray): Resulting histogram.
        """
        histogram = np.sum(opened[round(opened.shape[0]//2.5):, :],
                           axis=0)  # Histogram of the bottom half
        midpoint = np.int32(histogram.shape[0]/2)
        leftxbase = np.argmax(histogram[:midpoint])
        rightxbase = np.argmax(histogram[midpoint:]) + midpoint
        return histogram

    def lane_search(self, opened: np.ndarray, histogram: np.ndarray, nwindows: int = 10, offset: int = 15, minpix: int = 10, lane_type: str = "combined", window_debug: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Searching for lane with sliding windows.

        Args:
            opened (np.ndarray): Input frame. Has to be a binary image.
            histogram (np.ndarray): The binary input frames histogram.
            nwindows (int, optional): Number of sliding windows vertically for each lane (right and left). Defaults to 10.
            offset (int, optional): Horizontal width of the rectangle from the middle point. Defaults to 15.
            minpix (int, optional): Minimum number of pixels in a sliding window to recognize as a lane piece. Defaults to 50.

        Returns:
            _type_: _description_
        """
        midpoint = np.int32(
            histogram.shape[0]/2)  # .shape[0] for histogram is width
        # leftxbase = sc.find_peaks(histogram[:midpoint])[0]
        # rightxbase = sc.find_peaks(histogram[midpoint:])[0]
        # try:
        #     leftxbase = max(leftxbase) # Right atmost lane is the correct one
        #     rightxbase = min(rightxbase) + midpoint
        # except:
        #     try:
        #         rightxbase = min(rightxbase) + midpoint
        #         leftxbase = 0
        #     except:
        #         rightxbase = 0
        #         leftxbase = 0
        leftxbase = np.argmax(histogram[:midpoint])
        rightxbase = np.argmax(histogram[midpoint:]) + midpoint
        window_h = np.int32(opened.shape[0] / nwindows)  # window height
        nonzero = opened.nonzero()  # get white pixels
        nonzeroy = np.array(nonzero[0])  # y coordinates of white pixels
        nonzerox = np.array(nonzero[1])  # x coordinates of white pixels
        # where is currently the maximum number of white pixels on the left side
        leftx_current = leftxbase
        # where is currently the maximum number of white pixels on the left side
        rightx_current = rightxbase
        left_lane_inds = []  # Empty arrays for window contained pixels
        right_lane_inds = []
        nwindow_ok_left = 0
        nwindow_ok_right = 0
        nwindow_yellow_left = 0
        nwindow_yellow_right = 0
        window_proportion_to_yellow = round(nwindows/3)
        window_proportion_to_poly_fit = round(nwindows/3)
        left_side_ok = False
        right_side_ok = False
        yellow_lanes = False

        # Looping through windows
        for window in range(nwindows):
            # lower and upper edge of the window
            win_y_low = opened.shape[0] - (window + 1) * window_h
            win_y_high = opened.shape[0] - window * window_h
            # left and right coordinate of the window on the left side of the image
            win_xleft_left = leftx_current - offset
            win_xleft_right = leftx_current + offset
            # left and right coordinate of the window on the right side of the image
            win_xright_left = rightx_current - offset
            win_xright_right = rightx_current + offset

            if window_debug:
                # Making the windows for left and right side of the image
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
            if len(good_left) > minpix:
                if lane_type == "yellow":
                    nwindow_yellow_left += 1
                leftx_current = np.int32(np.mean(nonzerox[good_left]))
                nwindow_ok_left += 1
            if len(good_right) > minpix:
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
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_points = [leftx, lefty]
        right_points = [rightx, righty]

        if nwindow_yellow_left > window_proportion_to_yellow and nwindow_yellow_right > window_proportion_to_yellow:
            yellow_lanes = True

        if nwindow_ok_left > window_proportion_to_poly_fit:
            left_side_ok = True
        if nwindow_ok_right > window_proportion_to_poly_fit:
            right_side_ok = True

        sides_ok = [left_side_ok, right_side_ok]

        return left_points, right_points, sides_ok, yellow_lanes

    # Apply 2nd degree polynomial fit to lane pixels to fit curves
    def poly_fit(self, left, right, left_fit_prev, right_fit_prev, nwindow_ok):
        leftx, lefty = left[:]
        rightx, righty = right[:]
        left_side_ok, right_side_ok = nwindow_ok[:]

        if left_side_ok and right_side_ok:
            left_fit = np.polyfit(lefty, leftx, 1)
            right_fit = np.polyfit(righty, rightx, 1)
            return left_fit, right_fit
        elif left_side_ok and ~right_side_ok:
            left_fit = np.polyfit(lefty, leftx, 1)
            return left_fit, right_fit_prev,
        elif ~left_side_ok and right_side_ok:
            right_fit = np.polyfit(righty, rightx, 1)
            return left_fit_prev, right_fit
        else:
            return left_fit_prev, right_fit_prev

        # try:
        #     left_fit = np.polyfit(lefty, leftx, 1)
        #     right_fit = np.polyfit(righty, rightx, 1)
        #     print("Both lanes found")
        #     return left_fit, right_fit, yellow_lanes
        # except:
        #     try:
        #         left_fit = left_fit_prev
        #         right_fit = np.polyfit(righty, rightx, 1)
        #         print("Left lane not found")
        #         return left_fit, right_fit, yellow_lanes
        #     except:
        #         try:
        #             left_fit = np.polyfit(lefty, leftx, 1)
        #             right_fit = right_fit_prev
        #             print("Right lane not found")
        #             return left_fit, right_fit, yellow_lanes
        #         except:
        #             left_fit = left_fit_prev
        #             right_fit = right_fit_prev
        #             print("No lanes found")
        #             return left_fit, right_fit, yellow_lanes

    # Visually show detected lanes area
    def draw_lane_lines(self, frame: np.ndarray, birdseye: np.ndarray, Minv: np.ndarray, discarded: np.ndarray, left_fit: np.ndarray, right_fit: np.ndarray):
        """Draw the lane lines and the lane on the input frame.

        Args:
            frame (np.ndarray): _description_
            birdseye (np.ndarray): _description_
            Minv (np.ndarray): _description_
            discarded (np.ndarray): _description_
            left_fit (np.ndarray): _description_
            right_fit (np.ndarray): _description_
            CAMERA_OFFSET (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
        left_fitx = left_fit[0]*ploty + left_fit[1]
        right_fitx = right_fit[0]*ploty + right_fit[1]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array(
            [np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        warp_zero = np.zeros_like(birdseye).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        cv.fillPoly(color_warp, np.int_([pts]), (0, 120, 0))
        cv.fillPoly(color_warp, np.int_([pts_mean]), (255, 0, 0))

        newwarp = cv.warpPerspective(
            color_warp, Minv, (frame.shape[1], frame.shape[0]))
        final = cv.addWeighted(frame, 1, newwarp, 0.5, 0)
        original = np.concatenate((discarded, final), axis=0)

        lookaway_distance = int(pts_mean.shape[1]/2)
        # shape: (1, 230, 2); second -1 index indicating the most farthest point the detector can show
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
