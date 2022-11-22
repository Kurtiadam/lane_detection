import unittest
import numpy as np
from frame import FrameDB
from resmod import ResMod


class FrameDimensions(unittest.TestCase):
    def testFrameDimensions(self):
        self.frame = FrameDB('example_material\lane_video.mkv')
        self.streaming, self.base = self.frame.import_frame()
        self.assertEqual(np.size(self.base), 1080*1920*3)
        "Unit test for the individual frames if they are 1920*1080*3 \
            (3 channels)"

    def testDownscaleAndCroppingDimensions(self):
        self.frame = FrameDB('example_material\lane_video.mkv')
        self.streaming, self.base = self.frame.import_frame()
        self.res_mod = ResMod()
        self.downscaled = self.res_mod.downscale(self.base)
        self.assertEqual(np.size(self.downscaled), 640*480*3)

        self.ds_cr, self.discarded = self.res_mod.extract_roi(self.downscaled)
        self.assertEqual(np.size(self.ds_cr), (self.downscaled.shape[0]-self.res_mod.CROP_VERT_START)*(self.downscaled.shape[1]-self.res_mod.CROP_HOR_START)*3)
        "Unit test for the downscaling and cropping functions"


if __name__ == '__main__':
    unittest.main()
