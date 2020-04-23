import unittest
from torchvision import set_video_backend
import test_io


set_video_backend('video_reader')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(test_io)
    unittest.TextTestRunner(verbosity=1).run(suite)
