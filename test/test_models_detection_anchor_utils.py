import torch
import unittest
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList


class Tester(unittest.TestCase):
    def test_incorrect_anchors(self):
        incorrect_sizes = ((2, 4, 8), (32, 8), )
        incorrect_aspects = (0.5, 1.0)
        anc = AnchorGenerator(incorrect_sizes, incorrect_aspects)
        image1 = torch.randn(3, 800, 800)
        image_list = ImageList(image1, [(800, 800)])
        feature_maps = [torch.randn(1, 50)]
        self.assertRaises(ValueError, anc, image_list, feature_maps)
