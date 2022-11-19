import numpy as np
from constants import CropConst

class Crop:
    def __init__(self, crop_consts: CropConst = CropConst) -> None:
        self.UPPER_WHITE = crop_consts.UPPER_WHITE.value
        self.UPPER_YELLOW = crop_consts.UPPER_YELLOW.value
        self.LOWER_WHITE = crop_consts.LOWER_WHITE.value
        self.LOWER_YELLOW = crop_consts.LOWER_YELLOW.value