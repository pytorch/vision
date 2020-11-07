import unittest
from torchvision.models.detection.anchor_utils import AnchorGenerator


class Tester(unittest.TestCase):
    def test_incorrect_anchors(self):
        incorrect_sizes = (128, 258, 512)
        incorrect_aspects = (0.5, 1.0, 2.0)
        self.assertRaises(ValueError, AnchorGenerator, incorrect_sizes, incorrect_aspects)


if __name__ == '__main__':
    unittest.main()
