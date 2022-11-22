from enum import Enum
import numpy as np

class CropConst(Enum):
    """Constants for the cropping function"""
    LOWER_YELLOW = np.array([25,70,100])
    UPPER_YELLOW = np.array([70,200,255])
    LOWER_WHITE = np.array([27,122,25])
    UPPER_WHITE = np.array([101,255,150])