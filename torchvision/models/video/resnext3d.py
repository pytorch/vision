# Modified from
# https://github.com/facebookresearch/ClassyVision/blob/309d4f12431c6b4d8540010a781dc2aa25fe88e7/classy_vision/models/r2plus1_util.py
# https://github.com/facebookresearch/ClassyVision/blob/309d4f12431c6b4d8540010a781dc2aa25fe88e7/classy_vision/models/resnext3d_block.py
# https://github.com/facebookresearch/ClassyVision/blob/309d4f12431c6b4d8540010a781dc2aa25fe88e7/classy_vision/models/resnext3d_stage.py
# https://github.com/facebookresearch/ClassyVision/blob/309d4f12431c6b4d8540010a781dc2aa25fe88e7/classy_vision/models/resnext3d_stem.py
# https://github.com/facebookresearch/ClassyVision/blob/309d4f12431c6b4d8540010a781dc2aa25fe88e7/classy_vision/models/resnext3d.py

from collections import OrderedDict
import torch.nn as nn
from ....utils import _log_api_usage_once
from typing import Optional, List

_skip_transformations = {
    "postactivated_shortcut": PostactivatedShortcutTransformation,
    "preactivated_shortcut": PreactivatedShortcutTransformation,
    # For more types of skip transformations, add them below
}

_model_stems = {
    "r2plus1d_stem": R2Plus1DStem,
    "resnext3d_stem": ResNeXt3DStem,
    # For more types of model stem, add them below
}

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


residual_transformations = {
    "basic_r2plus1d_transformation": BasicR2Plus1DTransformation,
    "basic_transformation": BasicTransformation,
    "postactivated_bottleneck_transformation": PostactivatedBottleneckTransformation,
    "preactivated_bottleneck_transformation": PreactivatedBottleneckTransformation,
    # For more types of residual transformations, add them below
}

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
        skip_transformation_type: str,
        residual_transformation_type: str,
        num_groups: int = 1,
        inplace_relu: bool = True,
        bn_eps: float = 1e-5,
        bn_mmt: float = 0.1,
        disable_pre_activation: bool = False,
    ):
        super().__init__()

        assert skip_transformation_type in _skip_transformations, (
            "unknown skip transformation: %s" % skip_transformation_type
        )

        if (dim_in != dim_out) or (spatial_stride != 1) or (temporal_stride != 1):
            self.skip = _skip_transformations[skip_transformation_type](
                dim_in,
                dim_out,
                temporal_stride,
                spatial_stride,
                bn_eps=bn_eps,
                bn_mmt=bn_mmt,
                disable_pre_activation=disable_pre_activation,
            )

        assert residual_transformation_type in residual_transformations, (
            "unknown residual transformation: %s" % residual_transformation_type
        )
        self.residual = residual_transformations[residual_transformation_type](
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
        skip_transformation_type: str,
        residual_transformation_type: str,
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
                    skip_transformation_type,
                    residual_transformation_type,
                    num_groups=num_groups[p],
                    inplace_relu=inplace_relu,
                    bn_eps=bn_eps,
                    bn_mmt=bn_mmt,
                    disable_pre_activation=block_disable_pre_activation,
                )
                block_name = self._block_name(p, stage_idx, i)
                blocks.append((block_name, res_block))

            if final_stage and (
                residual_transformation_type == "preactivated_bottleneck_transformation"
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
        self, temporal_kernel, spatial_kernel, input_planes, stem_planes
    ):
        super().__init__()
        self._construct_stem(
            temporal_kernel, spatial_kernel, input_planes, stem_planes
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



