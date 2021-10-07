import unittest

import test_io
from torchvision import set_video_backend  # noqa: 401


# Disabling the video backend switching temporarily
# set_video_backend('video_reader')


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(test_io)
    unittest.TextTestRunner(verbosity=1).run(suite)
