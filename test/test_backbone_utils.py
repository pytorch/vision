import unittest


import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class ResnetFPNBackboneTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32

    def test_resnet18_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet18_fpn = resnet_fpn_backbone(backbone_name='resnet18', pretrained=False)
        y = resnet18_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])

    def test_resnet50_fpn_backbone(self):
        device = torch.device('cpu')
        x = torch.rand(1, 3, 300, 300, dtype=self.dtype, device=device)
        resnet50_fpn = resnet_fpn_backbone(backbone_name='resnet50', pretrained=False)
        y = resnet50_fpn(x)
        self.assertEqual(list(y.keys()), ['0', '1', '2', '3', 'pool'])
