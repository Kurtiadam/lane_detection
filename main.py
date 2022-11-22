from frame import FrameDB
from resmod import ResMod
from visualizer import Visualizer
import cv2 as cv
import numpy as np


class LaneDetection:
    """Class for running the lane detection algorithm"""

    def __init__(self, frame: FrameDB = FrameDB('example_material\lane_video.mkv'), res_mod: ResMod = ResMod(), visualizer: Visualizer = Visualizer()):
        """Initializing input classes

        Args:
            frame (FrameDB): Class for handling the example video file
            res_mod (ResMod): Class for resolution modifying methods
            visualizer (Visualizer): Class for visualizing the results
        """
        self.frame = frame
        self.res_mod = res_mod
        self.visualizer = visualizer

    def run(self) -> None:
        """Running the the stream of the video file"""
        while (self.frame.streaming):
            self.streaming, imp_frame = self.frame.import_frame()
            downscaled = self.res_mod.downscale(imp_frame)
            ds_cr, disc = self.res_mod.extract_roi(downscaled)
            self.visualizer.render(ds_cr)
            # Stopping streaming with q key
            if cv.waitKey(1) == ord('q'):
                break
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    lane_detector = LaneDetection(frame=FrameDB(
        'example_material\lane_video.mkv'), res_mod=ResMod(), visualizer=Visualizer())
    lane_detector.run()


if __name__ == "__main__":
    main()
