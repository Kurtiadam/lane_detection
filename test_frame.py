import unittest
import numpy as np
from frame import FrameDB

class FrameDimensions(unittest.TestCase):
        def testFrameDimensions(self):
            self.frame = FrameDB('example_material\lane_video.mkv')
            self.streaming, self.base  = self.frame.import_frame()
            self.assertEqual(np.size(self.base),1080*1920*3)

if __name__ == '__main__':
    unittest.main()