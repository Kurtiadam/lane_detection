import unittest
import numpy as np
from frame import FrameDB

class FrameDimensions(unittest.TestCase):
        def testFrameDimensions(self):
            self.frame = FrameDB('example_material\lane_video.mkv')
            self.streaming, self.base  = self.frame.import_frame()
            self.assertEqual(np.size(self.base),1080*1920*3)
            "Unit test for the individual frames if they are 1920*1080*3 \
            (3 channels)"

if __name__ == '__main__':
    unittest.main()