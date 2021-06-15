import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import pytest


def test_lol():
    if torch.cuda.is_available():
        raise ValueError("Yes")
    else:
        raise ValueError("Nope")

@pytest.mark.parametrize('backbone_name', ('resnet18', 'resnet50'))
def test_resnet_fpn_backbone(backbone_name):
    x = torch.rand(1, 3, 300, 300, dtype=torch.float32, device='cpu')
    y = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=False)(x)
    assert list(y.keys()) == ['0', '1', '2', '3', 'pool']
