from enum import Enum
import numpy as np


class ResModConst(Enum):
    """Constants for the downscaling and cropping function"""
    CROP_VERT_START = 250
    CROP_HOR_START = 0


class ColorSpaceTransConst(Enum):
    """Constants for the HLS color space function"""
    # LOWER_YELLOW = np.array([25,70,100])
    # UPPER_YELLOW = np.array([70,200,255])
    # LOWER_WHITE = np.array([27,122,25])
    # UPPER_WHITE = np.array([101,255,150])
