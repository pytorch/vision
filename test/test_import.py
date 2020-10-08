import torch, torchvision
import unittest


class ImportTester(unittest.TestCase):
    def test__C(self):
        self.assertTrue(torchvision._HAS_OPS)

    def test_image(self):
        torch.ops.load_library(torchvision.io.image.ext_specs.origin)

    def test_video(self):
        import torchvision.io._video_opt as vopt
        torch.ops.load_library(vopt.ext_specs.origin)


if __name__ == '__main__':
    unittest.main()
