from frame import FrameDB
from preproc import Preproc
from visualizer import Visualizer
from feat_ext import FeatExtract
import cv2 as cv
import numpy as np
import time


class LaneDetection:
    """Class for running the lane detection algorithm."""

    def __init__(self, frame: FrameDB = FrameDB(), prepoc: Preproc = Preproc(), featext: FeatExtract = FeatExtract(), visualizer: Visualizer = Visualizer()):
        """Initializing input classes.

        Args:
            frame (FrameDB): Class for handling the example video file.
            prepoc (Preproc): Class for image processing methods.
            featext (FeatExtract): Class for feature extraction.
            visualizer (Visualizer): Class for visualizing results.
        """
        self.frame = frame
        self.prepoc = prepoc
        self.featext = featext
        self.visualizer = visualizer

        self.mod_frame_array = []
        self.left_poly = [0, 0]
        self.right_poly = [0, 0]
        self.yellow_lanes_flag = False
        self.prev_time = 1

        # Constants for preprocessing
        # Color space transform constants [Hue (0-180), Lightness (0-255), Saturation (0-255)]
        self.LOWER_YELLOW = np.array([15, 80, 100])
        self.UPPER_YELLOW = np.array([30, 200, 255])
        self.LOWER_WHITE = np.array([0, 120, 0])
        self.UPPER_WHITE = np.array([180, 255, 255])

        self.DOWNSCALE_TARGET_RES = [640, 480]
        self.CROP_VERT_START = 250
        self.CROP_HOR_START = 0
        self.PERS_TRANS_LEFTUPPER = [self.DOWNSCALE_TARGET_RES[0]/2-150/2, 40]
        self.PERS_TRANS_RIGHTUPPER = [self.DOWNSCALE_TARGET_RES[0]/2+150/2, 40]
        self.PERS_TRANS_LEFTLOWER = [self.DOWNSCALE_TARGET_RES[0]/2-400/2, 230]
        self.PERS_TRANS_RIGHTLOWER = [
            self.DOWNSCALE_TARGET_RES[0]/2+400/2, 230]

        self.SOBEL_THRESH_LOW = 80

        # Constants for feature extraction
        # Proportion of ROI to non-ROI of histograms vertical axis. (histogram height / HISTOGRAM_ROI_PROP)
        self.HISTOGRAM_ROI_PROP = 2.5
        self.NWINDOWS = 20
        self.WINDOW_HOR_OFFSET = 25
        self.MINPIX = 1

    def run(self, birdseye_view__points_debug: bool = False, histogram_debug: bool = False, sliding_windows_debug: bool = False, save_result: bool = False) -> None:
        while (self.frame.streaming):
            try:
                # FPS measurement
                current_time = time.time()
                fps = int(1/(current_time-self.prev_time))
                self.prev_time = current_time

                # Image acquisition
                self.streaming, imp_frame = self.frame.import_frame()

                # Preprocessing
                downscaled = self.prepoc.downscale(
                    imp_frame, self.DOWNSCALE_TARGET_RES)
                downscaled_cropped, disc = self.prepoc.extract_roi(
                    downscaled, self.CROP_VERT_START, self.CROP_HOR_START)
                combined_lanes_binary, yellow_lanes_binary = self.prepoc.colorspace_transform(
                    downscaled_cropped, self.LOWER_YELLOW, self.UPPER_YELLOW, self.LOWER_WHITE, self.UPPER_WHITE)

                # White AND yellow preprocessing and lane detection
                if not self.yellow_lanes_flag:
                    birdseye_binary, Minv, birdview_points = self.prepoc.birdseye_transform(
                        combined_lanes_binary, self.PERS_TRANS_LEFTUPPER, self.PERS_TRANS_RIGHTUPPER, self.PERS_TRANS_LEFTLOWER, self.PERS_TRANS_RIGHTLOWER)
                    sobel_binary = self.prepoc.make_binary(
                        birdseye_binary, self.SOBEL_THRESH_LOW)
                    opened = self.prepoc.opening(sobel_binary)
                    histogram = self.featext.make_histogram(
                        opened, self.HISTOGRAM_ROI_PROP)
                    left, right, nwindow_ok, self.yellow_lanes_flag = self.featext.lane_search(
                        opened, histogram, "combined", self.NWINDOWS, self.WINDOW_HOR_OFFSET, self.MINPIX, window_debug=sliding_windows_debug)
                    self.left_poly, self.right_poly = self.featext.poly_fit(
                        left, right, self.left_poly, self.right_poly, nwindow_ok)
                    direction, original = self.featext.draw_lane_lines(
                        downscaled_cropped, sobel_binary, Minv, disc, self.left_poly, self.right_poly)

                    if sliding_windows_debug:
                        # self.visualizer.render_cv(birdseye_binary, 'Bird')
                        # self.visualizer.render_cv(sobel_binary, 'Sobel')
                        self.visualizer.render_cv(
                            opened, 'Sliding windows debugging')

                # Only yellow preprocessing and lane detection
                birdseye_binary_yellow, Minv, birdview_points = self.prepoc.birdseye_transform(
                    yellow_lanes_binary, self.PERS_TRANS_LEFTUPPER, self.PERS_TRANS_RIGHTUPPER, self.PERS_TRANS_LEFTLOWER, self.PERS_TRANS_RIGHTLOWER)
                sobel_binary_yellow = self.prepoc.make_binary(
                    birdseye_binary_yellow, self.SOBEL_THRESH_LOW)
                opened_yellow = self.prepoc.opening(sobel_binary_yellow)
                histogram_yellow = self.featext.make_histogram(
                    opened_yellow, self.HISTOGRAM_ROI_PROP)
                left, right, nwindow_ok, self.yellow_lanes_flag = self.featext.lane_search(
                    opened_yellow, histogram_yellow, "yellow", self.NWINDOWS, self.WINDOW_HOR_OFFSET, self.MINPIX, window_debug=sliding_windows_debug)

                # If yellow lanes have been found on both sides of the vehicle
                if self.yellow_lanes_flag:
                    self.left_poly, self.right_poly = self.featext.poly_fit(
                        left, right, self.left_poly, self.right_poly, nwindow_ok)
                    direction_yellow, original_yellow = self.featext.draw_lane_lines(
                        downscaled_cropped, sobel_binary_yellow, Minv, disc, self.left_poly, self.right_poly)
                    direction = direction_yellow
                    original = original_yellow

                    if sliding_windows_debug:
                        # self.visualizer.render_cv(birdseye_binary_yellow, 'Bird')
                        # self.visualizer.render_cv(sobel_binary_yellow, 'Sobel')
                        self.visualizer.render_cv(
                            opened_yellow, 'Sliding windows debugging')

                # Final output frame notation and visualization
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
                if cv.waitKey(10) == ord('q'):
                    break
            except Exception as error:
                print(error)
                # If the program runs into an error it saves the output if saving is enabled
                if save_result:
                    self.frame.save_to_database(self.mod_frame_array)
                self.frame.cap.release()
                cv.destroyAllWindows()
        # If the program is closed by pressing the 'q' key, the output will be save if saving is enabled
        if save_result:
            self.frame.save_to_database(self.mod_frame_array)
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    # Initializing and running the lane detection algorithm. Change between different options for running the program here: (True = ON)
    # mode: 0: lane detection based on color filtered frames; 1: lane detection based on the lightness channel
    # birdseye_view__points_debug: Visualizing the points which the perspective transform is based on
    # histogram_debug: While enabled, by pressing the 'h' key, the user can see the given binary frame's histogram
    # sliding_windows_debug: Lets the user inspect the workings of the sliding windows technique
    # save_result: saves the output video with the detected lanes into a new folder (Can take longer)
    lane_detector = LaneDetection(frame=FrameDB(
        'example_material\\example_video.mp4'), prepoc=Preproc(mode = 0), featext=FeatExtract(), visualizer=Visualizer())
    lane_detector.run(birdseye_view__points_debug=False,
                      histogram_debug=True, sliding_windows_debug=True, save_result=False)


if __name__ == "__main__":
    main()
