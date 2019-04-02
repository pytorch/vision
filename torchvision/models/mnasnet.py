import math

import torch
import torch.nn as nn

# Paper suggests 0.9997 momentum, for TensFlow. Equivalent PyTorch
# momentum is 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997

class _InvertedResidual(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 expansion_factor: int, bn_momentum: float = 0.1) -> None:
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.apply_residual:
            return self.layers.forward(input) + input
        else:
            return self.layers.forward(input)


def _stack(in_ch: int, out_ch: int, kernel_size: int, stride: int,
          exp_factor: int, repeats: int, bn_momentum: float) -> nn.Sequential:
    """ Creates a stack of inverted residuals as seen in e.g. MobileNetV2 or
    MNasNet. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                             bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                             bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int,
                          round_up_bias: float = 0.9) -> int:
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _scale_depths(depths: List[int], alpha: float) -> List[int]:
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNasNet(torch.nn.Module):
    """ MNasNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    >>> model = MNasNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model.forward(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """

    def __init__(self, num_classes: int, alpha: float, dropout:float=0.2) -> None:
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNasNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[5], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        ]
        self.layers = nn.Sequential(*layers)
        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(inplace=True, p=0.2), nn.Linear(1280, self.num_classes))
        else:
            self.classifier = nn.Linear(1280, self.num_classes)

        self._initialize_weights()

    def features(self, x):
        return self.layers.forward(x).squeeze()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MNasNet0_5(MNasNet):
    """ MNasNet with depth multiplier of 0.5. """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, 0.5)

class MNasNet0_75(MNasNet):
    """ MNasNet with depth multiplier of 0.75. """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, 0.75)

class MNasNet1_0(MNasNet):
    """ MNasNet with depth multiplier of 1.0. """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, 1.0)


class MNasNet1_3(MNasNet):
    """ MNasNet with depth multiplier of 1.3. """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, 1.3)


