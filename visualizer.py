import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Visualizer():
    @staticmethod
    def render_cv(frame: np.ndarray, title: str = "Output"):
        """This function shows the image with opencv in a new window to the user.

        Args:
            frame (np.ndarray): frame (np.ndarray): Frame to show with opencv.
            title (str, optional): Window name. Defaults to "Output".
        """
        cv.imshow(title, frame)

    @staticmethod
    def plot_birdview(frame: np.ndarray, points=[]):
        """This function shows the image with matplotlib in a new window to the user. Runs only once.

        Args:
            frame (np.ndarray): Frame to show with matplotlib.
            points (list, optional): Points to plot on the image. Defaults to [].
        """
        if getattr(Visualizer.plot_birdview, 'has_run', False):
            return
        Visualizer.plot_birdview.has_run = True

        leftupper, rightupper, leftlower, rightlower = [
            points[i] for i in (0, 1, 2, 3)]
        plt.plot(leftupper[0], leftupper[1], marker="o", color="red")
        plt.plot(rightupper[0], rightupper[1], marker="o", color="red")
        plt.plot(leftlower[0], leftlower[1], marker="o", color="red")
        plt.plot(rightlower[0], rightlower[1], marker="o", color="red")
        plt.imshow(frame)
        plt.show()

    @staticmethod
    def plot_histogram(histogram: np.ndarray):
        """This function shows the histogram with matplotlib in a new window to the user.

        Args:
            histogram (np.ndarray): Input histogram returned by make_histogram method.
        """
        plt.xlabel("X coordinates")
        plt.ylabel("Number of white pixels")
        plt.plot(histogram)
        plt.show()

    @staticmethod
    def write_text(img: np.ndarray, direction: str, yellow_lanes: bool, fps: int) -> np.ndarray:
        """Signaling the driver to take the appropiate wheel operation.

        Args:
            img (np.ndarray): Input frame.
            direction (str): Direction of the deviation from the lane center.

        Returns:
            np.ndarray: Marked frame.
        """
        final = cv.putText(img, 'FPS: '+str(fps), (500, 470), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
        if yellow_lanes:
            final = cv.putText(img, 'FOLLOWING YELLOW LANES', (5, 470), cv.FONT_HERSHEY_COMPLEX, 1, (0,215, 255), 1, cv.LINE_AA)
        match direction:
            case 0:
                final = cv.putText(img, 'IN CENTER!', (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,255, 0), 2, cv.LINE_AA)
            case 1:
                direction_text = "WARNING! KEEP LEFT!"
                final = cv.putText(img, direction_text, (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0, 255), 2, cv.LINE_AA)
            case -1:
                direction_text = "WARNING! KEEP RIGHT!"
                final = cv.putText(img, direction_text, (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0, 255), 2, cv.LINE_AA)
        return final
