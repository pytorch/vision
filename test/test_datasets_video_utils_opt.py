import unittest
from torchvision import set_video_backend
import test_datasets_video_utils


set_video_backend('video_reader')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(test_datasets_video_utils)
    unittest.TextTestRunner(verbosity=1).run(suite)
