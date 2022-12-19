from frame import FrameDB
from preproc import Preproc
from visualizer import Visualizer
from feat_ext import FeatExtract
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection:
    """Class for running the lane detection algorithm"""

    def __init__(self, frame: FrameDB = FrameDB('example_material\lane_video.mkv'), prepoc: Preproc = Preproc(), featext: FeatExtract = FeatExtract(), visualizer: Visualizer = Visualizer()):
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

    def run(self, birdseye_view__points_debug: bool, histogram_debug: bool) -> None:
        while (self.frame.streaming):
            self.streaming, imp_frame = self.frame.import_frame()
            downscaled = self.prepoc.downscale(imp_frame)
            downscaled_cropped, disc = self.prepoc.extract_roi(downscaled)
            extracted_lanes, yellow_lanes = self.prepoc.extract_lanes(downscaled_cropped, lower_yellow=np.array(
                [15, 80, 50]), upper_yellow=np.array([30, 200, 255]), lower_white=np.array([0, 100, 0]), upper_white=np.array([180, 255, 255]))
            birdseye, Minv, birdview_points = self.prepoc.birdseye_transform(
                extracted_lanes)
            blurred, binary = self.prepoc.make_binary(birdseye)
            opened = self.prepoc.opening(binary)
            histogram = self.featext.make_histogram(opened)
            left_poly, right_poly = self.featext.lane_search(
                opened, histogram, nwindows=15, offset=10)
            self.visualizer.render_cv(opened, 'Opened')
            self.visualizer.render_cv(downscaled_cropped, 'Downscaled')
            if birdseye_view__points_debug:
                self.visualizer.plot_birdview(extracted_lanes, birdview_points)
            elif histogram_debug:
                if cv.waitKey(1) == ord('h'):
                    self.visualizer.plot_histogram(histogram)
            # Stopping streaming with q key
            if cv.waitKey(1) == ord('q'):
                break
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    lane_detector = LaneDetection(frame=FrameDB(
        'example_material\lane_video.mkv'), prepoc=Preproc(), featext=FeatExtract(), visualizer=Visualizer())
    lane_detector.run(birdseye_view__points_debug=False, histogram_debug=True)


if __name__ == "__main__":
    main()
