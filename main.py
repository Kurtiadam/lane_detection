from frame import FrameDB
from preproc import Preproc
from visualizer import Visualizer
import cv2 as cv
import numpy as np


class LaneDetection:
    """Class for running the lane detection algorithm"""

    def __init__(self, frame: FrameDB = FrameDB('example_material\lane_video.mkv'), prepoc: Preproc = Preproc(), visualizer: Visualizer = Visualizer()):
        """Initializing input classes

        Args:
            frame (FrameDB): Class for handling the example video file
            prepoc (Preproc): Class for resolution modifying methods
            visualizer (Visualizer): Class for visualizing results
        """
        self.frame = frame
        self.prepoc = prepoc
        self.visualizer = visualizer

    def run(self) -> None:
        """Running the the stream of the video file"""
        while (self.frame.streaming):
            self.streaming, imp_frame = self.frame.import_frame()
            downscaled = self.prepoc.downscale(imp_frame)
            dowscaled_cropped, disc = self.prepoc.extract_roi(downscaled)
            extracted_lanes, yellow_lanes = self.prepoc.extract_lanes(dowscaled_cropped, lower_yellow = np.array([15,80,50]), upper_yellow = np.array([30,200,255]), lower_white = np.array([0,100,0]), upper_white = np.array([180,255,255]))
            self.visualizer.render(extracted_lanes)
            # Stopping streaming with q key
            if cv.waitKey(1) == ord('q'):
                break
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    lane_detector = LaneDetection(frame=FrameDB(
        'example_material\lane_video.mkv'), prepoc=Preproc(), visualizer=Visualizer())
    lane_detector.run()


if __name__ == "__main__":
    main()
