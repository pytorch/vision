# Modified from ClassyVision Resnext3d

from ast import Call
from collections import OrderedDict
from torch import Tensor, nn
from ...utils import _log_api_usage_once
from typing import Callable, Optional, List
from torchvision.models._api import WeightsEnum
from typing import Any


__all__ = [
    "ResNeXt3D",
    "resnext3d_postact_r2d101",
    "resnext3d_preact_r2d50",
    "resnext3d_pretact_r2d101",
    "resnext3d_postact_r2d50"
]

def _r2plus1_unit(
    dim_in: int,
    dim_out: int,
    temporal_stride: int,
    spatial_stride: int,
    groups: int,
    inplace_relu: bool,
    bn_eps: float,
    bn_mmt: float,
    dim_mid: Optional[int]=None,
):
    """
    Implementation of `R(2+1)D unit <https://arxiv.org/abs/1711.11248>`_.
    Decompose one 3D conv into one 2D spatial conv and one 1D temporal conv.
    Choose the middle dimensionality so that the total No. of parameters
    in 2D spatial conv and 1D temporal conv is unchanged.
    """
    if dim_mid is None:
        dim_mid = int(dim_out * dim_in * 3 * 3 * 3 / (dim_in * 3 * 3 + dim_out * 3))
    # 1x3x3 group conv, BN, ReLU
    conv_middle = nn.Conv3d(
        dim_in,
        dim_mid,
        [1, 3, 3],  # kernel
        stride=[1, spatial_stride, spatial_stride],
        padding=[0, 1, 1],
        groups=groups,
        bias=False,
    )
    conv_middle_bn = nn.BatchNorm3d(dim_mid, eps=bn_eps, momentum=bn_mmt)
    conv_middle_relu = nn.ReLU(inplace=inplace_relu)
    # 3x1x1 group conv
    conv = nn.Conv3d(
        dim_mid,
        dim_out,
        [3, 1, 1],  # kernel
        stride=[temporal_stride, 1, 1],
        padding=[1, 0, 0],
        groups=groups,
        bias=False,
    )
    return nn.Sequential(conv_middle, conv_middle_bn, conv_middle_relu, conv)


class BasicTransformation(nn.Module):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride: int,
        spatial_stride: int,
        groups: int,
        inplace_relu: bool=True,
        bn_eps:float=1e-5,
        bn_mmt:float=0.1,
    ):
        super().__init__()
        # 3x3x3 group conv, BN, ReLU.
        branch2a = nn.Conv3d(
            dim_in,
            dim_out,
            [3, 3, 3],  # kernel
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=[1, 1, 1],
            groups=groups,
            bias=False,
        )
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)
        # 3x3x3 group conv, BN, ReLU.
        branch2b = nn.Conv3d(
            dim_out,
            dim_out,
            [3, 3, 3],  # kernel
            stride=[1, 1, 1],
            padding=[1, 1, 1],
            groups=groups,
            bias=False,
        )
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True

        self.transform = nn.Sequential(
            branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn
        )
        
    def forward(self, x):
        return self.transform(x)


class BasicR2Plus1DTransformation(nn.Module):
    """
    Basic transformation: 3x3x3 group conv, 3x3x3 group conv
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride: int,
        spatial_stride: int,
        groups: int,
        inplace_relu: bool=True,
        bn_eps: float=1e-5,
        bn_mmt: float=0.1,
    ):
        super().__init__()
        # Implementation of R(2+1)D operation <https://arxiv.org/abs/1711.11248>.
        # decompose the original 3D conv into one 2D spatial conv and one
        # 1D temporal conv
        branch2a = _r2plus1_unit(
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            groups,
            inplace_relu,
            bn_eps,
            bn_mmt,
        )
        branch2a_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2a_relu = nn.ReLU(inplace=inplace_relu)

        branch2b = _r2plus1_unit(
            dim_out,
            dim_out,
            1,  # temporal_stride
            1,  # spatial_stride
            groups,
            inplace_relu,
            bn_eps,
            bn_mmt,
        )
        branch2b_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        branch2b_bn.final_transform_op = True

        self.transform = nn.Sequential(
            branch2a, branch2a_bn, branch2a_relu, branch2b, branch2b_bn
        )

        def forward(self, x):
            return self.transform(x)


class PostactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride: int,
        spatial_stride: int,
        num_groups: int,
        dim_inner: int,
        temporal_kernel_size: int =3,
        temporal_conv_1x1: bool =True,
        spatial_stride_1x1:bool=False,
        inplace_relu: bool=True,
        bn_eps: float=1e-5,
        bn_mmt: float=0.1,
        **kwargs,
    ):
        super().__init__()
        temporal_kernel_size_1x1, temporal_kernel_size_3x3 = (temporal_kernel_size, 1) if temporal_conv_1x1 else (1, temporal_kernel_size)
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3.
        str1x1, str3x3 = (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        # Tx1x1 conv, BN, ReLU.
        self.branch2a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[temporal_kernel_size_1x1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0],
            bias=False,
        )
        self.branch2a_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2a_relu = nn.ReLU(inplace=inplace_relu)
        # Tx3x3 group conv, BN, ReLU.
        self.branch2b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [temporal_kernel_size_3x3, 3, 3],
            stride=[temporal_stride, str3x3, str3x3],
            padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups,
            bias=False,
        )
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        # 1x1x1 conv, BN.
        self.branch2c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.branch2c_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_bn.final_transform_op = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.branch2a(x)
        x = self.branch2a_bn(x)
        x = self.branch2a_relu(x)

        # Branch2b.
        x = self.branch2b(x)
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)

        # Branch2c
        x = self.branch2c(x)
        x = self.branch2c_bn(x)
        return x


class PreactivatedBottleneckTransformation(nn.Module):
    """
    Bottleneck transformation with pre-activation, which includes BatchNorm3D
        and ReLu. Conv3D kernsl are Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel (https://arxiv.org/abs/1603.05027).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride: int,
        spatial_stride: int,
        num_groups: int,
        dim_inner: int,
        temporal_kernel_size: int =3,
        temporal_conv_1x1: bool=True,
        spatial_stride_1x1:bool=False,
        inplace_relu:bool=True,
        bn_eps: float=1e-5,
        bn_mmt: float=0.1,
        disable_pre_activation: bool=False,
    ):
        super().__init__()
        (temporal_kernel_size_1x1, temporal_kernel_size_3x3) = (
            (temporal_kernel_size, 1)
            if temporal_conv_1x1
            else (1, temporal_kernel_size)
        )
        (str1x1, str3x3) = (
            (spatial_stride, 1) if spatial_stride_1x1 else (1, spatial_stride)
        )

        self.disable_pre_activation = disable_pre_activation
        if not disable_pre_activation:
            self.branch2a_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch2a_relu = nn.ReLU(inplace=inplace_relu)

        self.branch2a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[temporal_kernel_size_1x1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[temporal_kernel_size_1x1 // 2, 0, 0],
            bias=False,
        )
        # Tx3x3 group conv, BN, ReLU.
        self.branch2b_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2b_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [temporal_kernel_size_3x3, 3, 3],
            stride=[temporal_stride, str3x3, str3x3],
            padding=[temporal_kernel_size_3x3 // 2, 1, 1],
            groups=num_groups,
            bias=False,
        )
        # 1x1x1 conv, BN.
        self.branch2c_bn = nn.BatchNorm3d(dim_inner, eps=bn_eps, momentum=bn_mmt)
        self.branch2c_relu = nn.ReLU(inplace=inplace_relu)
        self.branch2c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.branch2c.final_transform_op = True

    def forward(self, x):
        # Branch2a
        if not self.disable_pre_activation:
            x = self.branch2a_bn(x)
            x = self.branch2a_relu(x)
        x = self.branch2a(x)
        # Branch2b
        x = self.branch2b_bn(x)
        x = self.branch2b_relu(x)
        x = self.branch2b(x)
        # Branch2c
        x = self.branch2c_bn(x)
        x = self.branch2c_relu(x)
        x = self.branch2c(x)
        return x



class PostactivatedShortcutTransformation(nn.Module):
    """
    Skip connection used in ResNet3D model.
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride: int,
        spatial_stride: int,
        bn_eps: float=1e-5,
        bn_mmt: float=0.1,
        **kwargs,
    ):
        super().__init__()
        # Use skip connection with projection if dim or spatial/temporal res change.
        assert (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1)
        self.branch1 = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=0,
            bias=False,
        )
        self.branch1_bn = nn.BatchNorm3d(dim_out, eps=bn_eps, momentum=bn_mmt)

    def forward(self, x):
        return self.branch1_bn(self.branch1(x))


class PreactivatedShortcutTransformation(nn.Module):
    """
    Skip connection with pre-activation, which includes BatchNorm3D and ReLU,
        in ResNet3D model (https://arxiv.org/abs/1603.05027).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        temporal_stride,
        spatial_stride,
        inplace_relu=True,
        bn_eps=1e-5,
        bn_mmt=0.1,
        disable_pre_activation=False,
        **kwargs
    ):
        super(PreactivatedShortcutTransformation, self).__init__()
        # Use skip connection with projection if dim or spatial/temporal res change.
        assert (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1)
        if not disable_pre_activation:
            self.branch1_bn = nn.BatchNorm3d(dim_in, eps=bn_eps, momentum=bn_mmt)
            self.branch1_relu = nn.ReLU(inplace=inplace_relu)
        self.branch1 = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=1,
            stride=[temporal_stride, spatial_stride, spatial_stride],
            padding=0,
            bias=False,
        )

    def forward(self, x):
        if hasattr(self, "branch1_bn") and hasattr(self, "branch1_relu"):
            x = self.branch1_relu(self.branch1_bn(x))
        x = self.branch1(x)
        return x

class ResBlock(nn.Module):
    """
    ResBlock class constructs redisual blocks. More details can be found in:
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_inner: int,
        temporal_kernel_size: int,
        temporal_conv_1x1: bool,
        temporal_stride: int,
        spatial_stride: int,
        skip_transformation: Callable[..., nn.Module],
        residual_transformation: Callable[..., nn.Module],
        num_groups: int = 1,
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
        disable_pre_activation: bool = False,
    ):
        super().__init__()

        if (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1):
            self.skip = skip_transformation(
                dim_in,
                dim_out,
                temporal_stride,
                spatial_stride,
                bn_eps=bn_eps,
                bn_mmt=bn_mmt,
                disable_pre_activation=disable_pre_activation,
            )

    
        self.residual = residual_transformation(
            dim_in,
            dim_out,
            temporal_stride,
            spatial_stride,
            num_groups,
            dim_inner,
            temporal_kernel_size=temporal_kernel_size,
            temporal_conv_1x1=temporal_conv_1x1,
            disable_pre_activation=disable_pre_activation,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        if hasattr(self, "skip"):
            x = self.skip(x) + self.residual(x)
        else:
            x = x + self.residual(x)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, SlowOnly), and multi-pathway (SlowFast) cases.
        More details can be found here:
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        stage_idx: List[int],
        dim_in: List[int],
        dim_out: List[int],
        dim_inner: List[int],
        temporal_kernel_basis: List[int],
        temporal_conv_1x1: List[bool],
        temporal_stride: List[int],
        spatial_stride: List[int],
        num_blocks: List[int],
        num_groups: List[int],
        skip_transformation: Callable[..., nn.Module],
        residual_transformation: Callable[..., nn.Module],
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
        disable_pre_activation: bool = False,
        final_stage: bool = False,
    ):
        """
        ResStage builds p streams, where p can be greater or equal to one.
        """
        super().__init__()

        if len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temporal_kernel_basis),
                    len(temporal_conv_1x1),
                    len(temporal_stride),
                    len(spatial_stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                }
            ) != 1:
            raise ValueError("The following arguments should have equal legth: dim_in, dim_out, temporal_kernel_basis, temporal_conv_1x1, temporal_stride, spatial_stride, num_blocks, dim_inner, num_groups")

        self.stage_idx = stage_idx
        self.num_blocks = num_blocks
        self.num_pathways = len(self.num_blocks)

        self.temporal_kernel_sizes = [
            (temporal_kernel_basis[i] * num_blocks[i])[: num_blocks[i]]
            for i in range(len(temporal_kernel_basis))
        ]

        for p in range(self.num_pathways):
            blocks = []
            for i in range(self.num_blocks[p]):
                # Retrieve the transformation function.
                # Construct the block.
                block_disable_pre_activation = (
                    True if disable_pre_activation and i == 0 else False
                )
                res_block = ResBlock(
                    dim_in[p] if i == 0 else dim_out[p],
                    dim_out[p],
                    dim_inner[p],
                    self.temporal_kernel_sizes[p][i],
                    temporal_conv_1x1[p],
                    temporal_stride[p] if i == 0 else 1,
                    spatial_stride[p] if i == 0 else 1,
                    skip_transformation,
                    residual_transformation,
                    num_groups=num_groups[p],
                    inplace_relu=inplace_relu,
                    bn_eps=bn_eps,
                    bn_mmt=bn_mmt,
                    disable_pre_activation=block_disable_pre_activation,
                )
                block_name = self._block_name(p, stage_idx, i)
                blocks.append((block_name, res_block))

            if final_stage and (
                isinstance(residual_transformation, PreactivatedBottleneckTransformation)
            ):
                # For pre-activation residual transformation, we conduct
                # activation in the final stage before continuing forward pass
                # through the head
                activate_bn = nn.BatchNorm3d(dim_out[p])
                activate_relu = nn.ReLU(inplace=True)
                activate_bn_name = "-".join([block_name, "bn"])
                activate_relu_name = "-".join([block_name, "relu"])
                blocks.append((activate_bn_name, activate_bn))
                blocks.append((activate_relu_name, activate_relu))

            self.add_module(self._pathway_name(p), nn.Sequential(OrderedDict(blocks)))


    def _block_name(self, pathway_idx, stage_idx, block_idx):
        return "pathway{}-stage{}-block{}".format(pathway_idx, stage_idx, block_idx)

    def _pathway_name(self, pathway_idx):
        return "pathway{}".format(pathway_idx)

    def forward(self, inputs):
        output = []
        for p in range(self.num_pathways):
            x = inputs[p]
            pathway_module = getattr(self, self._pathway_name(p))
            output.append(pathway_module(x))
        return output



class ResNeXt3DStemSinglePathway(nn.Module):
    """
    ResNe(X)t 3D basic stem module. Assume a single pathway.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: List[int],
        stride: List[int],
        padding: int,
        maxpool: bool = True,
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
    ):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt
        self.maxpool = maxpool

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.maxpool:
            x = self.pool_layer(x)
        return x


class R2Plus1DStemSinglePathway(ResNeXt3DStemSinglePathway):
    """
    R(2+1)D basic stem module. Assume a single pathway.
    Performs spatial convolution, temporal convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: List[int],
        stride: List[int],
        padding: int,
        maxpool: bool = True,
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
    ):
        super().__init__(
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            maxpool=maxpool,
            inplace_relu=inplace_relu,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )

    def _construct_stem(self, dim_in, dim_out):
        if self.stride[1] != self.stride[2]:
            raise ValueError("Only support identical height stride and width stride")

        self.conv = r2plus1_unit(
            dim_in,
            dim_out,
            self.stride[0],  # temporal_stride
            self.stride[1],  # spatial_stride
            1,  # groups
            self.inplace_relu,
            self.bn_eps,
            self.bn_mmt,
            dim_mid=45,  # hard-coded middle channels
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.maxpool:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )


class ResNeXt3DStemMultiPathway(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: List[int],
        stride: List[int],
        padding: List[int],
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1
    ):
        super().__init__()

        if len({len(dim_in), len(dim_out), len(kernel), len(stride), len(padding)}) != 1:
            raise ValueError("The following arguments should have equal legth: dim_in, dim_out, kernel, stride")
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.bn_eps = bn_eps
        self.bn_mmt = bn_mmt

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        self.blocks = {}
        for p in range(len(dim_in)):
            stem = ResNeXt3DStemSinglePathway(
                dim_in[p],
                dim_out[p],
                self.kernel[p],
                self.stride[p],
                self.padding[p],
                inplace_relu=self.inplace_relu,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt,
            )
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem

    def _stem_name(self, path_idx):
        return "stem-path{}".format(path_idx)

    def forward(self, x):
        for p in range(len(x)):
            stem_name = self._stem_name(p)
            x[p] = self.blocks[stem_name](x[p])
        return x


class R2Plus1DStemMultiPathway(ResNeXt3DStemMultiPathway):
    """
    Video R(2+1)D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: int,
        stride: List[int],
        padding: List[int],
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
    ):
        super().__init__(
            dim_in,
            dim_out,
            kernel,
            stride,
            padding,
            inplace_relu=inplace_relu,
            bn_eps=bn_eps,
            bn_mmt=bn_mmt,
        )

    def _construct_stem(self, dim_in, dim_out):
        self.blocks = {}
        for p in range(len(dim_in)):
            stem = R2Plus1DStemSinglePathway(
                dim_in[p],
                dim_out[p],
                self.kernel[p],
                self.stride[p],
                self.padding[p],
                inplace_relu=self.inplace_relu,
                bn_eps=self.bn_eps,
                bn_mmt=self.bn_mmt,
            )
            stem_name = self._stem_name(p)
            self.add_module(stem_name, stem)
            self.blocks[stem_name] = stem


class ResNeXt3DStem(nn.Module):
    def __init__(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        super().__init__()
        self._construct_stem(
            temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
        )

    def _construct_stem(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        self.stem = ResNeXt3DStemMultiPathway(
            [input_planes],
            [stem_planes],
            [[temporal_kernel, spatial_kernel, spatial_kernel]],
            [[1, 2, 2]],  # stride
            [
                [temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]
            ],  # padding
            
        )

    def forward(self, x):
        return self.stem(x)


class R2Plus1DStem(ResNeXt3DStem):
    def __init__(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        super(R2Plus1DStem, self).__init__(
            temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
        )

    def _construct_stem(
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes, maxpool
    ):
        self.stem = R2Plus1DStemMultiPathway(
            [input_planes],
            [stem_planes],
            [[temporal_kernel, spatial_kernel, spatial_kernel]],
            [[1, 2, 2]],  # stride
            [
                [temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2]
            ],  # padding
            maxpool=[maxpool],
        )
class FullyConvolutionalLinear(nn.Module):
    def __init__(self, dim_in, num_classes):
        super(FullyConvolutionalLinear, self).__init__()
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        self.act = nn.Softmax(dim=4)
        

    def forward(self, x):
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.flatten(start_dim=1)
        return x


class FullyConvolutionalLinearHead(nn.Module):
    """
    This head defines a 3d average pooling layer (:class:`torch.nn.AvgPool3d` or
    :class:`torch.nn.AdaptiveAvgPool3d` if pool_size is None) followed by a fully
    convolutional linear layer. This layer performs a fully-connected projection
    during training, when the input size is 1x1x1.
    It performs a convolutional projection during testing when the input size
    is larger than 1x1x1.
    """

    def __init__(
        self,
        num_classes: int,
        in_plane: int,
        pool_size: Optional[List[int]],
        use_dropout: Optional[bool] = None,
        dropout_ratio: float = 0.5,
    ):
        """
        Constructor for FullyConvolutionalLinearHead.
        Args:
            unique_id: A unique identifier for the head. Multiple instances of
                the same head might be attached to a model, and unique_id is used
                to refer to them.
            num_classes: Number of classes for the head.
            in_plane: Input size for the fully connected layer.
            pool_size: Optional kernel size for the 3d pooling layer. If None, use
                :class:`torch.nn.AdaptiveAvgPool3d` with output size (1, 1, 1).
            use_dropout: Whether to apply dropout after the pooling layer.
            dropout_ratio: dropout ratio.
        """
        super().__init__()
        if pool_size is not None:
            self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        else:
            self.final_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_ratio)
        # we separate average pooling from the fully-convolutional linear projection
        # because for multi-path models such as SlowFast model, the input can be
        # more than 1 tesnor. In such case, we can define a new head to combine multiple
        # tensors via concat or addition, do average pooling, but still reuse
        # FullyConvolutionalLinear inside of it.
        self.head_fcl = FullyConvolutionalLinear(
            in_plane, num_classes
        )
        
    
    def forward(self, x):
        out = self.final_avgpool(x)
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.head_fcl(out)
        return out

class ResNeXt3D(nn.Module):
    """
    Implementation of:
        1. Conventional `post-activated 3D ResNe(X)t <https://arxiv.org/
        abs/1812.03982>`_.
        2. `Pre-activated 3D ResNe(X)t <https://arxiv.org/abs/1811.12814>`_.
        The model consists of one stem, a number of stages, and one or multiple
        heads that are attached to different blocks in the stage.
    """

    def __init__(
        self,
        input_planes,
        clip_crop_size,
        skip_transformation,
        residual_transformation: Callable[..., nn.Module],
        frames_per_clip,
        num_blocks,
        stem: Callable[..., nn.Module],
        stem_planes,
        stem_temporal_kernel,
        stem_spatial_kernel,
        stem_maxpool,
        stage_planes,
        stage_temporal_kernel_basis,
        temporal_conv_1x1,
        stage_temporal_stride,
        stage_spatial_stride,
        num_groups,
        width_per_group,
        zero_init_residual_transform,
    ):
        """
        Args:
            input_planes (int): the channel dimension of the input. Normally 3 is used
                for rgb input.
            clip_crop_size (int): spatial cropping size of video clip at train time.
            skip_transformation (Callable[..., nn.Module]): the skip transformation.
            residual_transformation (Callable[.., Module]): the residual transformation.
            frames_per_clip (int): Number of frames in a video clip.
            num_blocks (list): list of the number of blocks in stages.
            stem (Callable[..., nn.Module]): stem block.
            stem_planes (int): the output dimension of the convolution in the model
                stem.
            stem_temporal_kernel (int): the temporal kernel size of the convolution
                in the model stem.
            stem_spatial_kernel (int): the spatial kernel size of the convolution
                in the model stem.
            stem_maxpool (bool): If true, perform max pooling.
            stage_planes (int): the output channel dimension of the 1st residual stage
            stage_temporal_kernel_basis (list): Basis of temporal kernel sizes for
                each of the stage.
            temporal_conv_1x1 (bool): Only useful for BottleneckTransformation.
                In a pathaway, if True, do temporal convolution in the first 1x1
                Conv3d. Otherwise, do it in the second 3x3 Conv3d.
            stage_temporal_stride (int): the temporal stride of the residual
                transformation.
            stage_spatial_stride (int): the spatial stride of the the residual
                transformation.
            num_groups (int): number of groups for the convolution.
                num_groups = 1 is for standard ResNet like networks, and
                num_groups > 1 is for ResNeXt like networks.
            width_per_group (int): Number of channels per group in 2nd (group)
                conv in the residual transformation in the first stage
            zero_init_residual_transform (bool): if true, the weight of last
                operation, which could be either BatchNorm3D in post-activated
                transformation or Conv3D in pre-activated transformation, in the
                residual transformation is initialized to zero
        """
        super().__init__()

        self.input_planes = input_planes
        self.clip_crop_size = clip_crop_size
        self.frames_per_clip = frames_per_clip
        self.num_blocks = num_blocks

        self.stem = stem(
            stem_temporal_kernel,
            stem_spatial_kernel,
            input_planes,
            stem_planes,
            stem_maxpool,
        )


        num_stages = len(num_blocks)
        out_planes = [stage_planes * 2 ** i for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [
            num_groups * width_per_group * 2 ** i for i in range(num_stages)
        ]

        stages = []
        for s in range(num_stages):
            stage = ResStage(
                s + 1,  # stem is viewed as stage 0, and following stages start from 1
                [in_planes[s]],
                [out_planes[s]],
                [inner_planes[s]],
                [stage_temporal_kernel_basis[s]],
                [temporal_conv_1x1[s]],
                [stage_temporal_stride[s]],
                [stage_spatial_stride[s]],
                [num_blocks[s]],
                [num_groups],
                skip_transformation,
                residual_transformation,
                disable_pre_activation=(s == 0),
                final_stage=(s == (num_stages - 1)),
            )
            stages.append(stage)

        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_residual_transform)
        self.head = FullyConvolutionalLinearHead(
            num_classes = 1000,
            in_plane = 2048,
            pool_size = [1, 7, 7],
            use_dropout=False,
        )

    def _init_parameter(self, zero_init_residual_transform):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if (
                    hasattr(m, "final_transform_op")
                    and m.final_transform_op
                    and zero_init_residual_transform
                ):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if (
                    hasattr(m, "final_transform_op")
                    and m.final_transform_op
                    and zero_init_residual_transform
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor):
        """
        Args:
            x (torch.Tensor): video input with shape N x C x T x H x W.
        """
        # single pathway input (make list out of tensor input)
        out = self.stem([x])
        out = self.stages(out)
        # single pathway output get first element from the list before head
        out = self.head(out[0])
        
        return out

class ResNeXt3D_PreAct_r2d50_Weights(WeightsEnum):
    pass
class ResNeXt3D_PreAct_r2d101_Weights(WeightsEnum):
    pass
class ResNeXt3D_PostAct_r2d50_Weights(WeightsEnum):
    pass

class ResNeXt3D_PostAct_r2d101_Weights(WeightsEnum):
    pass


def resnext3d_preact_r2d50(*, weights: Optional[ResNeXt3D_PreAct_r2d50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNeXt3D:
    """
    Constructs a single pathway resnext3d_preact_r2d50 architecture.
    Args:
        weights (:class:`~fb.models.ResNeXt3D_PreAct_r2d50_Weights`, optional): The pretrained
            weights to use. Currently no pretrained weights are provided. 
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``fb.models.resnext3d.ResNeXt3D``
    """
    return ResNeXt3D(
            input_planes = 3,
            clip_crop_size = 224,
            skip_transformation = PreactivatedShortcutTransformation,
            residual_transformation = PreactivatedBottleneckTransformation,
            frames_per_clip = 1,
            num_blocks = [3, 4, 6, 3],
            stem = ResNeXt3DStem,
            stem_planes = 32,
            stem_temporal_kernel = 1,
            stem_spatial_kernel = 5,
            stem_maxpool = True,
            stage_planes = 256,
            stage_temporal_kernel_basis = [[1], [1], [1], [1]],
            temporal_conv_1x1 = [True, True, True, True],
            stage_temporal_stride = [1, 1, 1, 1],
            stage_spatial_stride = [1, 2, 2, 2],
            num_groups = 1,
            width_per_group = 64,
            zero_init_residual_transform = True,
            **kwargs,
        )


def resnext3d_pretact_r2d101(*, weights: Optional[ResNeXt3D_PreAct_r2d101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNeXt3D:
    """
    Constructs a single pathway resnext3d_pretact_r2d101 architecture.
    Args:
        weights (:class:`~fb.models.ResNeXt3D_PreAct_r2d101_Weights`, optional): The pretrained
            weights to use. Currently no pretrained weights are provided. 
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``fb.models.resnext3d.ResNeXt3D``
    """
    return ResNeXt3D(
            input_planes = 3,
            clip_crop_size = 224,
            skip_transformation = PreactivatedShortcutTransformation,
            residual_transformation = PreactivatedBottleneckTransformation,
            frames_per_clip = 1,
            num_blocks = [3, 4, 23, 3],
            stem = ResNeXt3DStem,
            stem_planes = 32,
            stem_temporal_kernel = 1,
            stem_spatial_kernel = 5,
            stem_maxpool = True,
            stage_planes = 256,
            stage_temporal_kernel_basis = [[1], [1], [1], [1]],
            temporal_conv_1x1 = [True, True, True, True],
            stage_temporal_stride = [1, 1, 1, 1],
            stage_spatial_stride = [1, 2, 2, 2],
            num_groups = 1,
            width_per_group = 64,
            zero_init_residual_transform = True,
            **kwargs,
    )


def resnext3d_postact_r2d50(*, weights: Optional[ResNeXt3D_PostAct_r2d50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNeXt3D:
    """
    Constructs a single pathway resnext3d_postact_r2d50 architecture.
    Args:
        weights (:class:`~fb.models.ResNeXt3D_PostAct_r2d50_Weights`, optional): The pretrained
            weights to use. Currently no pretrained weights are provided. 
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``fb.models.resnext3d.ResNeXt3D``
    """
    return ResNeXt3D(
            input_planes = 3,
            clip_crop_size = 224,
            skip_transformation = PostactivatedShortcutTransformation,
            residual_transformation = PostactivatedBottleneckTransformation,
            frames_per_clip = 1,
            num_blocks = [3, 4, 6, 3],
            stem = ResNeXt3DStem,
            stem_planes = 64,
            stem_temporal_kernel = 1,
            stem_spatial_kernel = 7,
            stem_maxpool = True,
            stage_planes = 256,
            stage_temporal_kernel_basis = [[1], [1], [1], [1]],
            temporal_conv_1x1 = [True, True, True, True],
            stage_temporal_stride = [1, 1, 1, 1],
            stage_spatial_stride = [1, 2, 2, 2],
            num_groups = 1,
            width_per_group = 64,
            zero_init_residual_transform = True,
            **kwargs,
    )




def resnext3d_postact_r2d101(*, weights: Optional[ResNeXt3D_PostAct_r2d101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNeXt3D:
    """
    Constructs a single pathway resnext3d_postact_r2d101 architecture.
    Args:
        weights (:class:`~fb.models.ResNeXt3D_PostAct_r2d101_Weights`, optional): The pretrained
            weights to use. Currently no pretrained weights are provided. 
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``fb.models.resnext3d.ResNeXt3D``
    """
    return ResNeXt3D(
            input_planes = 3,
            clip_crop_size = 224,
            skip_transformation = PostactivatedShortcutTransformation,
            residual_transformation = PostactivatedBottleneckTransformation,
            frames_per_clip = 1,
            num_blocks = [3, 4, 23, 3],
            stem = ResNeXt3DStem,
            stem_planes = 64,
            stem_temporal_kernel = 1,
            stem_spatial_kernel = 7,
            stem_maxpool = True,
            stage_planes = 256,
            stage_temporal_kernel_basis = [[1], [1], [1], [1]],
            temporal_conv_1x1 = [True, True, True, True],
            stage_temporal_stride = [1, 1, 1, 1],
            stage_spatial_stride = [1, 2, 2, 2],
            num_groups = 1,
            width_per_group = 64,
            zero_init_residual_transform = True,
            **kwargs,
        )