import copy
import torch
from torchvision.models.detection import _utils
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import unittest
from torchvision.models.detection import backbone_utils


class Tester(unittest.TestCase):
    def test_balanced_positive_negative_sampler(self):
        sampler = _utils.BalancedPositiveNegativeSampler(4, 0.25)
        # keep all 6 negatives first, then add 3 positives, last two are ignore
        matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
        pos, neg = sampler(matched_idxs)
        # we know the number of elements that should be sampled for the positive (1)
        # and the negative (3), and their location. Let's make sure that they are
        # there
        self.assertEqual(pos[0].sum(), 1)
        self.assertEqual(pos[0][6:9].sum(), 1)
        self.assertEqual(neg[0].sum(), 3)
        self.assertEqual(neg[0][0:6].sum(), 3)

    def test_resnet_fpn_backbone_frozen_layers(self):
        # we know how many initial layers and parameters of the network should
        # be frozen for each trainable_backbone_layers parameter value
        # i.e all 53 params are frozen if trainable_backbone_layers=0
        # ad first 24 params are frozen if trainable_backbone_layers=2
        expected_frozen_params = {0: 53, 1: 43, 2: 24, 3: 11, 4: 1, 5: 0}
        for train_layers, exp_froz_params in expected_frozen_params.items():
            model = backbone_utils.resnet_fpn_backbone(
                'resnet50', pretrained=False, trainable_layers=train_layers)
            # boolean list that is true if the param at that index is frozen
            is_frozen = [not parameter.requires_grad for _, parameter in model.named_parameters()]
            # check that expected initial number of layers are frozen
            self.assertTrue(all(is_frozen[:exp_froz_params]))

    def test_validate_resnet_inputs_detection(self):
        # default number of backbone layers to train
        ret = backbone_utils._validate_trainable_layers(
            pretrained=True, trainable_backbone_layers=None, max_value=5, default_value=3)
        self.assertEqual(ret, 3)
        # can't go beyond 5
        with self.assertRaises(AssertionError):
            ret = backbone_utils._validate_trainable_layers(
                pretrained=True, trainable_backbone_layers=6, max_value=5, default_value=3)
        # if not pretrained, should use all trainable layers and warn
        with self.assertWarns(UserWarning):
            ret = backbone_utils._validate_trainable_layers(
                pretrained=False, trainable_backbone_layers=0, max_value=5, default_value=3)
        self.assertEqual(ret, 5)

    def test_transform_copy_targets(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.rand(3, 200, 300), torch.rand(3, 200, 200)]
        targets = [{'boxes': torch.rand(3, 4)}, {'boxes': torch.rand(2, 4)}]
        targets_copy = copy.deepcopy(targets)
        out = transform(image, targets)  # noqa: F841
        self.assertTrue(torch.equal(targets[0]['boxes'], targets_copy[0]['boxes']))
        self.assertTrue(torch.equal(targets[1]['boxes'], targets_copy[1]['boxes']))

    def test_not_float_normalize(self):
        transform = GeneralizedRCNNTransform(300, 500, torch.zeros(3), torch.ones(3))
        image = [torch.randint(0, 255, (3, 200, 300), dtype=torch.uint8)]
        targets = [{'boxes': torch.rand(3, 4)}]
        with self.assertRaises(TypeError):
            out = transform(image, targets)  # noqa: F841


if __name__ == '__main__':
    unittest.main()
