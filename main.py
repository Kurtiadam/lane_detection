from frame import FrameDB
from preproc import Preproc
from visualizer import Visualizer
from feat_ext import FeatExtract
import cv2 as cv
import numpy as np
import time


class LaneDetection:
    """Class for running the lane detection algorithm"""

    def __init__(self, frame: FrameDB = FrameDB(), prepoc: Preproc = Preproc(), featext: FeatExtract = FeatExtract(), visualizer: Visualizer = Visualizer()):
        """Initializing input classes

        Args:
            frame (FrameDB): Class for handling the example video file
            prepoc (Preproc): Class for image processing methods
            featext (FeatExtract): Class for feature extraction
            visualizer (Visualizer): Class for visualizing results
        """
        self.frame = frame
        self.prepoc = prepoc
        self.featext = featext
        self.visualizer = visualizer

        self.mod_frame_array = []
        self.left_poly = [0, 0]
        self.right_poly = [0, 0]
        self.yellow_lanes_flag = False

        self.SOBEL_THRESH_LOW = 90
        self.DOWNSCALE_TARGET_RES = [640, 480]
        self.CROP_VERT_START = 250
        self.CROP_HOR_START = 0
        self.PERS_TRANS_LEFTUPPER = [self.DOWNSCALE_TARGET_RES[0]/2-150/2, 40]
        self.PERS_TRANS_RIGHTUPPER = [self.DOWNSCALE_TARGET_RES[0]/2+150/2, 40]
        self.PERS_TRANS_LEFTLOWER = [self.DOWNSCALE_TARGET_RES[0]/2-400/2, 230]
        self.PERS_TRANS_RIGHTLOWER = [
            self.DOWNSCALE_TARGET_RES[0]/2+400/2, 230]

        self.LOWER_YELLOW = np.array([15, 80, 100])
        self.UPPER_YELLOW = np.array([30, 200, 255])
        self.LOWER_WHITE = np.array([0, 110, 0])
        self.UPPER_WHITE = np.array([180, 255, 255])

    def run(self, mode: int = 0, birdseye_view__points_debug: bool = False, histogram_debug: bool = False, sliding_windows_debug: bool = False, save_result: bool = False) -> None:
        prev_time = 1
        while (self.frame.streaming):
            try:
                current_time = time.time()
                fps = int(1/(current_time-prev_time))
                prev_time = current_time
                self.streaming, imp_frame = self.frame.import_frame()
                downscaled = self.prepoc.downscale(
                    imp_frame, self.DOWNSCALE_TARGET_RES)
                downscaled_cropped, disc = self.prepoc.extract_roi(
                    downscaled, self.CROP_VERT_START, self.CROP_HOR_START)
                combined_lanes_binary, yellow_lanes_binary = self.prepoc.colorspace_transform(
                    downscaled_cropped, mode, self.LOWER_YELLOW, self.UPPER_YELLOW, self.LOWER_WHITE, self.UPPER_WHITE)

                # sobel_x_reg = cv.Sobel(combined_lanes_binary, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType = cv.BORDER_DEFAULT)
                # abs_sobel_x_reg = cv.convertScaleAbs(sobel_x_reg)
                # cv.imshow('Sobel regular', abs_sobel_x_reg)
                # sobel_reg_thresholded = np.zeros_like(abs_sobel_x_reg)
                # sobel_reg_thresholded[(abs_sobel_x_reg >= self.SOBEL_THRESH_LOW) & (abs_sobel_x_reg <= 255)] = 255
                # cv.imshow('Sobel regular thresholded', sobel_reg_thresholded)

                # White and yellow lane detection
                if not self.yellow_lanes_flag:
                    birdseye_binary, Minv, birdview_points, = self.prepoc.birdseye_transform(
                        combined_lanes_binary, self.PERS_TRANS_LEFTUPPER, self.PERS_TRANS_RIGHTUPPER, self.PERS_TRANS_LEFTLOWER, self.PERS_TRANS_RIGHTLOWER)
                    sobel_binary = self.prepoc.make_binary(
                        birdseye_binary, self.SOBEL_THRESH_LOW)
                    opened = self.prepoc.opening(sobel_binary)
                    histogram = self.featext.make_histogram(opened)
                    #print("Left: ", self.left_poly)
                    #print("Right: ", self.right_poly)
                    left, right, nwindow_ok, self.yellow_lanes_flag = self.featext.lane_search(
                        opened, histogram, nwindows=20, offset=25, minpix=1, lane_type="combined", window_debug=sliding_windows_debug)
                    self.left_poly, self.right_poly = self.featext.poly_fit(
                        left, right, self.left_poly, self.right_poly, nwindow_ok)
                    direction, original = self.featext.draw_lane_lines(
                        downscaled_cropped, sobel_binary, Minv, disc, self.left_poly, self.right_poly, CAMERA_OFFSET=0)

                    self.visualizer.render_cv(birdseye_binary, 'Bird')
                    self.visualizer.render_cv(sobel_binary, 'Sobel_binary')
                    # Morphed, binary birdseyeview frames
                    self.visualizer.render_cv(opened, 'Opened')

                # Yellow lane detection
                birdseye_binary_yellow, Minv, birdview_points, = self.prepoc.birdseye_transform(
                    yellow_lanes_binary, self.PERS_TRANS_LEFTUPPER, self.PERS_TRANS_RIGHTUPPER, self.PERS_TRANS_LEFTLOWER, self.PERS_TRANS_RIGHTLOWER)
                sobel_binary_yellow = self.prepoc.make_binary(
                    birdseye_binary_yellow, self.SOBEL_THRESH_LOW)
                opened_yellow = self.prepoc.opening(sobel_binary_yellow)
                histogram_yellow = self.featext.make_histogram(opened_yellow)
                #print("Left: ", self.left_poly)
                #print("Right: ", self.right_poly)
                left, right, nwindow_ok, self.yellow_lanes_flag = self.featext.lane_search(
                    opened_yellow, histogram_yellow, nwindows=20, offset=25, minpix=1, lane_type="yellow", window_debug=sliding_windows_debug)

                if self.yellow_lanes_flag:
                    self.left_poly, self.right_poly = self.featext.poly_fit(
                        left, right, self.left_poly, self.right_poly, nwindow_ok)
                    direction_yellow, original_yellow = self.featext.draw_lane_lines(
                        downscaled_cropped, sobel_binary_yellow, Minv, disc, self.left_poly, self.right_poly, CAMERA_OFFSET=0)
                    direction = direction_yellow
                    original = original_yellow
                    print("Change-over to yellow lanes!")
                    self.visualizer.render_cv(birdseye_binary_yellow, 'Bird')
                    self.visualizer.render_cv(
                        sobel_binary_yellow, 'Sobel_binary')
                    # Morphed, binary birdseyeview frames
                    self.visualizer.render_cv(opened_yellow, 'Opened')

                final = self.visualizer.write_text(
                    original, direction, self.yellow_lanes_flag, fps)
                self.visualizer.render_cv(final, 'Final')
                if save_result:
                    self.mod_frame_array = self.frame.reconsturct_store(final)
                if birdseye_view__points_debug:
                    self.visualizer.plot_birdview(
                        combined_lanes_binary, birdview_points)
                if histogram_debug:
                    if cv.waitKey(10) == ord('h'):
                        self.visualizer.plot_histogram(histogram)
                # Stopping streaming with q key
                if cv.waitKey(10) == ord('q'):
                    break
            except Exception as error:
                print(error)
                if save_result:
                    self.frame.save_to_database(self.mod_frame_array)
        if save_result:
            self.frame.save_to_database(self.mod_frame_array)
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    lane_detector = LaneDetection(frame=FrameDB(
        'example_material\\example_video.mp4'), prepoc=Preproc(), featext=FeatExtract(), visualizer=Visualizer())
    lane_detector.run(mode=1, birdseye_view__points_debug=False,
                      histogram_debug=True, sliding_windows_debug=True, save_result=True)


if __name__ == "__main__":
    main()
