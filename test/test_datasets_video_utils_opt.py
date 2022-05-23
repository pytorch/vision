import unittest

import test_datasets_video_utils
from torchvision import set_video_backend  # noqa: 401

# Disabling the video backend switching temporarily
# set_video_backend('video_reader')


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(test_datasets_video_utils)
    unittest.TextTestRunner(verbosity=1).run(suite)
