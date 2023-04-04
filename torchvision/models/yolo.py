from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


def _get_padding(kernel_size: int, stride: int) -> Tuple[int, nn.Module]:
    """Returns the amount of padding needed by convolutional and max pooling layers.

    Determines the amount of padding needed to make the output size of the layer the input size divided by the stride.
    The first value that the function returns is the amount of padding to be added to all sides of the input matrix
    (``padding`` argument of the operation). If an uneven amount of padding is needed in different sides of the input,
    the second variable that is returned is an ``nn.ZeroPad2d`` operation that adds an additional column and row of
    padding. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        kernel_size: Size of the kernel.
        stride: Stride of the operation.

    Returns:
        padding, pad_op: The amount of padding to be added to all sides of the input and an ``nn.Identity`` or
        ``nn.ZeroPad2d`` operation to add one more column and row of padding if necessary.
    """
    # The output size is generally (input_size + padding - max(kernel_size, stride)) / stride + 1 and we want to
    # make it equal to input_size / stride.
    padding, remainder = divmod(max(kernel_size, stride) - stride, 2)

    # If the kernel size is an even number, we need one cell of extra padding, on top of the padding added by MaxPool2d
    # on both sides.
    pad_op: nn.Module = nn.Identity() if remainder == 0 else nn.ZeroPad2d((0, 1, 0, 1))

    return padding, pad_op


def _create_activation_module(name: Optional[str]) -> nn.Module:
    """Creates a layer activation module given its type as a string.

    Args:
        name: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic", "linear",
            or "none".
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "mish":
        return Mish()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "logistic":
        return nn.Sigmoid()
    if name == "linear" or name == "none" or name is None:
        return nn.Identity()
    raise ValueError(f"Activation type `{name}´ is unknown.")


def _create_normalization_module(name: Optional[str], num_channels: int) -> nn.Module:
    """Creates a layer normalization module given its type as a string.

    Group normalization uses always 8 channels. The most common network widths are divisible by this number.

    Args:
        name: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        num_channels: The number of input channels that the module expects.
    """
    if name == "batchnorm":
        return nn.BatchNorm2d(num_channels, eps=0.001)
    if name == "groupnorm":
        return nn.GroupNorm(8, num_channels, eps=0.001)
    if name == "none" or name is None:
        return nn.Identity()
    raise ValueError(f"Normalization layer type `{name}´ is unknown.")


class Conv(nn.Module):
    """A convolutional layer with optional layer normalization and activation.

    If ``padding`` is ``None``, the module tries to add padding so much that the output size will be the input size
    divided by the stride. If the input size is not divisible by the stride, the output size will be rounded upwards.

    Args:
        in_channels: Number of input channels that the layer expects.
        out_channels: Number of output channels that the convolution produces.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        bias: If ``True``, adds a learnable bias to the output.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()

        if padding is None:
            padding, self.pad = _get_padding(kernel_size, stride)
        else:
            self.pad = nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = _create_normalization_module(norm, out_channels)
        self.act = _create_activation_module(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class MaxPool(nn.Module):
    """A max pooling layer with padding.

    The module tries to add padding so much that the output size will be the input size divided by the stride. If the
    input size is not divisible by the stride, the output size will be rounded upwards.
    """

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        padding, self.pad = _get_padding(kernel_size, stride)
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        return self.maxpool(x)


class RouteLayer(nn.Module):
    """A routing layer concatenates the output (or part of it) from given layers.

    Args:
        source_layers: Indices of the layers whose output will be concatenated.
        num_chunks: Layer outputs will be split into this number of chunks.
        chunk_idx: Only the chunks with this index will be concatenated.
    """

    def __init__(self, source_layers: List[int], num_chunks: int, chunk_idx: int) -> None:
        super().__init__()
        self.source_layers = source_layers
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

    def forward(self, outputs: List[Tensor]) -> Tensor:
        chunks = [torch.chunk(outputs[layer], self.num_chunks, dim=1)[self.chunk_idx] for layer in self.source_layers]
        return torch.cat(chunks, dim=1)


class ShortcutLayer(nn.Module):
    """A shortcut layer adds a residual connection from the source layer.

    Args:
        source_layer: Index of the layer whose output will be added to the output of the previous layer.
    """

    def __init__(self, source_layer: int) -> None:
        super().__init__()
        self.source_layer = source_layer

    def forward(self, outputs: List[Tensor]) -> Tensor:
        return outputs[-1] + outputs[self.source_layer]


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class ReOrg(nn.Module):
    """Re-organizes the tensor so that every square region of four cells is placed into four different channels.

    The result is a tensor with half the width and height, and four times as many channels.
    """

    def forward(self, x: Tensor) -> Tensor:
        tl = x[..., ::2, ::2]
        bl = x[..., 1::2, ::2]
        tr = x[..., ::2, 1::2]
        br = x[..., 1::2, 1::2]
        return torch.cat((tl, bl, tr, br), dim=1)


class BottleneckBlock(nn.Module):
    """A residual block with a bottleneck layer.

    Args:
        in_channels: Number of input channels that the block expects.
        out_channels: Number of output channels that the block produces.
        hidden_channels: Number of output channels the (hidden) bottleneck layer produces. By default the number of
            output channels of the block.
        shortcut: Whether the block should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        shortcut: bool = True,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.convs = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm),
            Conv(hidden_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm),
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        y = self.convs(x)
        return x + y if self.shortcut else y


class TinyStage(nn.Module):
    """One stage of the "tiny" network architecture from YOLOv4.

    Args:
        num_channels: Number of channels in the input of the stage. Partial output will have as many channels and full
            output will have twice as many channels.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        num_channels: int,
        activation: Optional[str] = "leaky",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        hidden_channels = num_channels // 2
        self.conv1 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.mix = Conv(num_channels, num_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        partial = torch.chunk(x, 2, dim=1)[1]
        y1 = self.conv1(partial)
        y2 = self.conv2(y1)
        partial_output = self.mix(torch.cat((y2, y1), dim=1))
        full_output = torch.cat((x, partial_output), dim=1)
        return partial_output, full_output


class CSPStage(nn.Module):
    """One stage of a Cross Stage Partial Network (CSPNet).

    Encapsulates a number of bottleneck blocks in the "fusion first" CSP structure.

    `Chien-Yao Wang et al. <https://arxiv.org/abs/1911.11929>`_

    Args:
        in_channels: Number of input channels that the CSP stage expects.
        out_channels: Number of output channels that the CSP stage produces.
        depth: Number of bottleneck blocks that the CSP stage contains.
        shortcut: Whether the bottleneck blocks should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 1,
        shortcut: bool = True,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        # Instead of splitting the N output channels of a convolution into two parts, we can equivalently perform two
        # convolutions with N/2 output channels.
        hidden_channels = out_channels // 2

        self.split1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        bottlenecks: List[nn.Module] = [
            BottleneckBlock(hidden_channels, hidden_channels, shortcut=shortcut, norm=norm, activation=activation)
            for _ in range(depth)
        ]
        self.bottlenecks = nn.Sequential(*bottlenecks)
        self.mix = Conv(hidden_channels * 2, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.bottlenecks(self.split1(x))
        y2 = self.split2(x)
        return self.mix(torch.cat((y1, y2), dim=1))


class ELANStage(nn.Module):
    """One stage of an Efficient Layer Aggregation Network (ELAN).

    `Chien-Yao Wang et al. <https://arxiv.org/abs/2211.04800>`_

    Args:
        in_channels: Number of input channels that the ELAN stage expects.
        out_channels: Number of output channels that the ELAN stage produces.
        hidden_channels: Number of output channels that the computational blocks produce. The default value is half the
            number of output channels of the block, as in YOLOv7-W6, but the value varies between the variants.
        split_channels: Number of channels in each part after splitting the input to the cross stage connection and the
            computational blocks. The default value is the number of hidden channels, as in all YOLOv7 backbones. Most
            YOLOv7 heads use twice the number of hidden channels.
        depth: Number of computational blocks that the ELAN stage contains. The default value is 2. YOLOv7 backbones use
            2 to 4 blocks per stage.
        block_depth: Number of convolutional layers in one computational block. The default value is 2. YOLOv7 backbones
            have two convolutions per block. YOLOv7 heads (except YOLOv7-X) have 2 to 8 blocks with only one convolution
            in each.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        split_channels: Optional[int] = None,
        depth: int = 2,
        block_depth: int = 2,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm)

        def block(in_channels: int, out_channels: int) -> nn.Module:
            convs = [conv3x3(in_channels, out_channels)]
            for _ in range(block_depth - 1):
                convs.append(conv3x3(out_channels, out_channels))
            return nn.Sequential(*convs)

        # Instead of splitting the N output channels of a convolution into two parts, we can equivalently perform two
        # convolutions with N/2 output channels. However, in many YOLOv7 architectures, the number of hidden channels is
        # not exactly half the number of output channels.
        if hidden_channels is None:
            hidden_channels = out_channels // 2

        if split_channels is None:
            split_channels = hidden_channels

        self.split1 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

        blocks = [block(split_channels, hidden_channels)]
        for _ in range(depth - 1):
            blocks.append(block(hidden_channels, hidden_channels))
        self.blocks = nn.ModuleList(blocks)

        total_channels = (split_channels * 2) + (hidden_channels * depth)
        self.mix = Conv(total_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [self.split1(x), self.split2(x)]
        x = outputs[-1]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return self.mix(torch.cat(outputs, dim=1))


class CSPSPP(nn.Module):
    """Spatial pyramid pooling module from the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, activation=activation, norm=norm)

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels),
            conv(out_channels, out_channels, kernel_size=3),
            conv(out_channels, out_channels),
        )
        self.conv2 = conv(in_channels, out_channels)

        self.maxpool1 = MaxPool(kernel_size=5, stride=1)
        self.maxpool2 = MaxPool(kernel_size=9, stride=1)
        self.maxpool3 = MaxPool(kernel_size=13, stride=1)

        self.mix1 = nn.Sequential(
            conv(4 * out_channels, out_channels),
            conv(out_channels, out_channels, kernel_size=3),
        )
        self.mix2 = Conv(2 * out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.maxpool2(x1)
        x4 = self.maxpool3(x1)
        y1 = self.mix1(torch.cat((x1, x2, x3, x4), dim=1))
        y2 = self.conv2(x)
        return self.mix2(torch.cat((y1, y2), dim=1))


class FastSPP(nn.Module):
    """Fast spatial pyramid pooling module from YOLOv5.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.maxpool = MaxPool(kernel_size=5, stride=1)
        self.mix = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.conv(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y4 = self.maxpool(y3)
        return self.mix(torch.cat((y1, y2, y3, y4), dim=1))


class YOLOV4TinyBackbone(nn.Module):
    """Backbone of the "tiny" network architecture from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        activation: Optional[str] = "leaky",
        normalization: Optional[str] = "batchnorm",
    ):
        super().__init__()

        def smooth(num_channels: int) -> nn.Module:
            return Conv(num_channels, num_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            conv_module = Conv(
                in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization
            )
            return nn.Sequential(OrderedDict([("downsample", conv_module), ("smooth", smooth(out_channels))]))

        def maxpool(out_channels: int) -> nn.Module:
            return nn.Sequential(
                OrderedDict(
                    [
                        ("pad", nn.ZeroPad2d((0, 1, 0, 1))),
                        ("maxpool", MaxPool(kernel_size=2, stride=2)),
                        ("smooth", smooth(out_channels)),
                    ]
                )
            )

        def stage(out_channels: int, use_maxpool: bool) -> nn.Module:
            if use_maxpool:
                downsample_module = maxpool(out_channels)
            else:
                downsample_module = downsample(out_channels // 2, out_channels)
            stage_module = TinyStage(out_channels, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([("downsample", downsample_module), ("stage", stage_module)]))

        stages = [
            Conv(in_channels, width, kernel_size=3, stride=2, activation=activation, norm=normalization),
            stage(width * 2, False),
            stage(width * 4, True),
            stage(width * 8, True),
            maxpool(width * 16),
        ]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.stages[0](x)
        c2, x = self.stages[1](c1)
        c3, x = self.stages[2](x)
        c4, x = self.stages[3](x)
        c5 = self.stages[4](x)
        return [c1, c2, c3, c4, c5]


class YOLOV4Backbone(nn.Module):
    """A backbone that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Typically ``(32, 64, 128, 256, 512, 1024)``. The P6 variant
            adds one more stage with 1024 channels.
        depths: Number of bottleneck layers at each network stage. Typically ``(1, 1, 2, 8, 8, 4)``. The P6 variant uses
            ``(1, 1, 3, 15, 15, 7, 7)``.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024),
        depths: Sequence[int] = (1, 1, 2, 8, 8, 4),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        if len(widths) != len(depths):
            raise ValueError("Width and depth has to be given for an equal number of stages.")

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: int, out_channels: int, depth: int) -> nn.Module:
            csp = CSPStage(
                out_channels,
                out_channels,
                depth=depth,
                shortcut=True,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("csp", csp),
                    ]
                )
            )

        convs = [conv3x3(in_channels, widths[0])] + [conv3x3(widths[0], widths[0]) for _ in range(depths[0] - 1)]
        self.stem = nn.Sequential(*convs)
        self.stages = nn.ModuleList(
            stage(in_channels, out_channels, depth)
            for in_channels, out_channels, depth in zip(widths[:-1], widths[1:], depths[1:])
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outputs: List[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class YOLOV5Backbone(nn.Module):
    """The Cross Stage Partial Network backbone from YOLOv5.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def downsample(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Module:
            return Conv(
                in_channels, out_channels, kernel_size=kernel_size, stride=2, activation=activation, norm=normalization
            )

        def stage(in_channels: int, out_channels: int, depth: int) -> nn.Module:
            csp = CSPStage(
                out_channels,
                out_channels,
                depth=depth,
                shortcut=True,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("csp", csp),
                    ]
                )
            )

        stages = [
            downsample(in_channels, width, kernel_size=6),
            stage(width, width * 2, depth),
            stage(width * 2, width * 4, depth * 2),
            stage(width * 4, width * 8, depth * 3),
            stage(width * 8, width * 16, depth),
        ]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.stages[0](x)
        c2 = self.stages[1](c1)
        c3 = self.stages[2](c2)
        c4 = self.stages[3](c3)
        c5 = self.stages[4](c4)
        return [c1, c2, c3, c4, c5]


class YOLOV7Backbone(nn.Module):
    """A backbone that corresponds to the W6 variant of the Efficient Layer Aggregation Network from YOLOv7.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Before the first stage there will be one extra split of
            spatial resolution by a ``ReOrg`` layer, producing ``in_channels * 4`` channels.
        depth: Number of computational blocks at each network stage. YOLOv7-W6 backbone uses 2.
        block_depth: Number of convolutional layers in one computational block. YOLOv7-W6 backbone uses 2.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 128, 256, 512, 768, 1024),
        depth: int = 2,
        block_depth: int = 2,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: int, out_channels: int) -> nn.Module:
            elan = ELANStage(
                out_channels,
                out_channels,
                depth=depth,
                block_depth=block_depth,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("elan", elan),
                    ]
                )
            )

        self.stem = nn.Sequential(*[ReOrg(), conv3x3(in_channels * 4, widths[0])])
        self.stages = nn.ModuleList(
            stage(in_channels, out_channels) for in_channels, out_channels in zip(widths[:-1], widths[1:])
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outputs: List[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
