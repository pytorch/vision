from collections import OrderedDict
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

    def _init_test_anchor_generator(self):
        anchor_sizes = tuple((x,) for x in [32, 64, 128])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return anchor_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ('0', torch.rand(2, 8, s0 // 4, s1 // 4)),
            ('1', torch.rand(2, 16, s0 // 8, s1 // 8)),
            ('2', torch.rand(2, 32, s0 // 16, s1 // 16)),
        ]
        features = OrderedDict(features)
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 16, 32)
        features = self.get_features(images)
        features = list(features.values())
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(images, features)

        # Compute target anchors numbers
        grid_sizes = [f.shape[-2:] for f in features]
        num_anchors_target = 0
        for sizes, num_anchors_per_loc in zip(grid_sizes, model.num_anchors_per_location()):
            num_anchors_target += sizes[0] * sizes[1] * num_anchors_per_loc

        self.assertEqual(len(anchors), 2)
        self.assertEqual(num_anchors_target, 126)
        self.assertEqual(tuple(anchors[0].shape), (num_anchors_target, 4))
        self.assertEqual(tuple(anchors[1].shape), (num_anchors_target, 4))
