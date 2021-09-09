from torch import nn, Tenspr

class RegNetParams:
    pass

class SqueezeExcitation(nn.Module):
    """
    Squeeze and excitation layer from 
    `"Squeeze-and-Excitation Networks" <https://arxiv.org/pdf/1709.01507>`_.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: Optional[int] = 16,
        reduced_channels: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_channels is defined,
        # neither both nor none of them
        assert bool(reduction_ratio) != bool(reduced_channels)

        if activation is None:
            activation = nn.ReLU()

        reduced_channels = (
            in_channels // reduction_ratio if reduced_channels is None else reduced_channels
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1, bias=True),
            activation,
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled

class RegNet(nn.Module):
    pass

def regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    pass
