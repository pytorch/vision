import torch
from common_utils import TestCase
from torchvision.models.detection.anchor_utils import AnchorGenerator, DBoxGenerator
from torchvision.models.detection.image_list import ImageList


class Tester(TestCase):
    def test_incorrect_anchors(self):
        incorrect_sizes = ((2, 4, 8), (32, 8), )
        incorrect_aspects = (0.5, 1.0)
        anc = AnchorGenerator(incorrect_sizes, incorrect_aspects)
        image1 = torch.randn(3, 800, 800)
        image_list = ImageList(image1, [(800, 800)])
        feature_maps = [torch.randn(1, 50)]
        self.assertRaises(ValueError, anc, image_list, feature_maps)

    def _init_test_anchor_generator(self):
        anchor_sizes = ((10,),)
        aspect_ratios = ((1,),)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return anchor_generator

    def _init_test_dbox_generator(self):
        size = 300
        feature_map_sizes = [1]
        aspect_ratios = [[2]]
        dbox_generator = DBoxGenerator(size, feature_map_sizes, aspect_ratios)

        return dbox_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 15, 15)
        features = self.get_features(images)
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(images, features)

        # Estimate the number of target anchors
        grid_sizes = [f.shape[-2:] for f in features]
        num_anchors_estimated = 0
        for sizes, num_anchors_per_loc in zip(grid_sizes, model.num_anchors_per_location()):
            num_anchors_estimated += sizes[0] * sizes[1] * num_anchors_per_loc

        anchors_output = torch.tensor([[-5., -5., 5., 5.],
                                       [0., -5., 10., 5.],
                                       [5., -5., 15., 5.],
                                       [-5., 0., 5., 10.],
                                       [0., 0., 10., 10.],
                                       [5., 0., 15., 10.],
                                       [-5., 5., 5., 15.],
                                       [0., 5., 10., 15.],
                                       [5., 5., 15., 15.]])

        self.assertEqual(num_anchors_estimated, 9)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(tuple(anchors[0].shape), (9, 4))
        self.assertEqual(tuple(anchors[1].shape), (9, 4))
        self.assertEqual(anchors[0], anchors_output)
        self.assertEqual(anchors[1], anchors_output)

    def test_dbox_generator(self):
        images = torch.zeros(2, 3, 15, 15)
        features = [torch.zeros(2, 8, 3, 3)]
        image_shapes = [i.shape[-2:] for i in images]
        images = ImageList(images, image_shapes)

        model = self._init_test_dbox_generator()
        model.eval()
        dboxes = model(images, features)

        dboxes_output = torch.tensor([
            [0.4650, 0.4650, 0.5350, 0.5350],
            [0.4488, 0.4488, 0.5512, 0.5512],
            [0.4505, 0.4753, 0.5495, 0.5247],
            [0.4753, 0.4505, 0.5247, 0.5495]
        ])

        tol = 0.0001
        self.assertEqual(len(dboxes), 2)
        self.assertEqual(tuple(dboxes[0].shape), (4, 4))
        self.assertEqual(tuple(dboxes[1].shape), (4, 4))
        self.assertTrue(dboxes[0].allclose(dboxes_output, atol=tol))
        self.assertTrue(dboxes[1].allclose(dboxes_output, atol=tol))
