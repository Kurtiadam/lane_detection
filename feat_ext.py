import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal as sc


class FeatExtract:
    def make_histogram(self, opened: np.ndarray) -> np.ndarray:
        """This function makes a histogram out of the inputted frame on which the user can see the vertical pixel count of each column of pixels.

        Args:
            opened (np.ndarray): Input frame.

        Returns:
            histogram (np.ndarray): Resulting histogram.
        """
        histogram = np.sum(opened[opened.shape[0]//2:,:], axis = 0) #Histogram of the bottom half
        midpoint = np.int32(histogram.shape[0]/2)
        leftxbase = np.argmax(histogram[:midpoint])
        rightxbase = np.argmax(histogram[midpoint:]) + midpoint
        return histogram

    def lane_search(self, opened: np.ndarray, histogram: np.ndarray, nwindows: int = 10, offset: int = 15, minpix: int = 10) -> tuple[np.ndarray, np.ndarray]:
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
        midpoint = np.int32(histogram.shape[0]/2) #.shape[0] for histogram is width
        leftxbase = sc.find_peaks(histogram[:midpoint])[0]
        rightxbase = sc.find_peaks(histogram[midpoint:])[0]
        try:
            leftxbase = max(leftxbase) # Right atmost lane is the correct one
            rightxbase = min(rightxbase) + midpoint
        except:
            try:
                rightxbase = min(rightxbase) + midpoint
                leftxbase = 0
            except:
                rightxbase = 0
                leftxbase = 0
        out_frame = np.dstack((opened, opened, opened)) * 255 # Img to draw on
        window_h = np.int32(opened.shape[0] / nwindows) # window height
        nonzero = opened.nonzero() # get white pixels
        nonzeroy = np.array(nonzero[0]) # y coordinates of white pixels
        nonzerox = np.array(nonzero[1]) # x coordinates of white pixels
        leftx_current = leftxbase # where is currently the maximum number of white pixels on the left side
        rightx_current = rightxbase # where is currently the maximum number of white pixels on the left side
        left_lane_inds = [] # Empty arrays for window contained pixels
        right_lane_inds = []
        
        # Looping though windows
        for window in range(nwindows):
            win_y_low = opened.shape[0] - (window + 1) * window_h # lower and upper edge of the window
            win_y_high = opened.shape[0] - window * window_h
            win_xleft_left = leftx_current - offset # left and right coordinate of the window on the left side of the image
            win_xleft_right = leftx_current + offset
            win_xright_left = rightx_current - offset # left and right coordinate of the window on the right side of the image
            win_xright_right = rightx_current + offset
            # Making the windows for left and right side of the image
            cv.rectangle(opened, (win_xleft_left, win_y_low), (win_xleft_right, win_y_high),(255,255,0), 1)
            cv.rectangle(opened, (win_xright_left,win_y_low), (win_xright_right,win_y_high),(255,255,0), 1)
            
            # Picking the white pixels which the windows contain on both sides
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_left) &  (nonzerox < win_xleft_right)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_left) &  (nonzerox < win_xright_right)).nonzero()[0]
            
            # Making sure the number of contained pixels exceeds the minimum threshold
            # If it does, then get the x position of the lane with the mean values for the next window adjusting
            if len(good_left) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right]))
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

        # Apply 2nd degree polynomial fit to lane pixels to fit curves
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            print("Both lanes found")
            return left_fit, right_fit
        except:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                print("Left lane not found")
                return 0, right_fit
            except:
                print("No lanes found")
            # # Visualisation of progress
            # ploty = np.linspace(0, opened.shape[0]-1, opened.shape[0])
            # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            # plt.plot(right_fitx, color = 'red')
            # plt.show()

            # out_frame[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255] #Left lane with red
            # out_frame[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] #Right lane with blue

            # #plt.imshow(out_frame)
            # #plt.plot(left_fitx,  ploty, color = 'yellow')
            # #plt.plot(right_fitx, ploty, color = 'yellow')
