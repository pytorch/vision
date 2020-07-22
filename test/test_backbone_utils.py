import unittest


import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import resnet50


class ResnetFPNBackboneTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32

    def test_resnet18_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet18_fpn = resnet_fpn_backbone(backbone='resnet18', pretrained=False)
        y = resnet18_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet50_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet50_fpn = resnet_fpn_backbone(backbone='resnet50', pretrained=False)
        y = resnet50_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet50_fpn_backbone_with_callable(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet50_fpn = resnet_fpn_backbone(backbone=resnet50, pretrained=False)
        y = resnet50_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet101_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet101_fpn = resnet_fpn_backbone(backbone='resnet101', pretrained=False)
        y = resnet101_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet_fpn_backbone_invalid_backbone_type(self):
        with self.assertRaises(Exception):
            resnet_fpn_backbone(backbone=1, pretrained=False)

    def test_resnet_fpn_backbone_invalid_model_name(self):
        with self.assertRaises(Exception):
            resnet_fpn_backbone(backbone='resnet20', pretrained=False)

    def test_resnet18_fpn_backbone_invalid_frozen_layers(self):
        with self.assertRaises(Exception):
            resnet_fpn_backbone(backbone='resnet18', pretrained=False, trainable_layers=6)


if __name__ == '__main__':
    unittest.main()
