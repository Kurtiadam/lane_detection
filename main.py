from frame import FrameDB
import cv2 as cv

class LaneDetection:
    """Class for running the lane detection algorithm"""
    def __init__(self, frame:FrameDB('example_material\lane_video.mkv')):
        """Initializing input classes

        Args:
            frame (FrameDB): Class for handling the example video file
        """
        self.frame = frame

    def run(self) -> None:
        """Running the the stream of the video file"""
        while(self.frame.streaming):
            self.streaming, imported_frame = self.frame.import_frame()
            # Stopping streaming with q key
            if cv.waitKey(1) == ord('q'):
                break
        self.frame.cap.release()
        cv.destroyAllWindows()


def main():
    lane_detector = LaneDetection(frame = FrameDB('example_material\lane_video.mkv'))
    lane_detector.run()


if __name__ == "__main__":
    main()