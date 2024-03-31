import os
from collections import defaultdict
from numbers import Number
from typing import Any, List

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torch.utils._pytree import tree_map

from torchvision.models._api import Weights

aten = torch.ops.aten
quantized = torch.ops.quantized


def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    elif hasattr(i, "weight"):
        return i.weight().shape
    else:
        raise ValueError(f"Unknown type {type(i)}")


def prod(x):
    res = 1
    for i in x:
        res *= i
    return res


def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops


def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop


def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


def quant_conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for quantized convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))

    return conv_flop_count(x_shape, w_shape, out_shape, transposed=False)


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def conv_backward_flop(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0

    if output_mask[0]:
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

    return flop_count


def scaled_dot_product_flash_attention_flop(inputs: List[Any], outputs: List[Any]):
    # FIXME: this needs to count the flops of this kernel
    # https://github.com/pytorch/pytorch/blob/207b06d099def9d9476176a1842e88636c1f714f/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp#L52-L267
    return 0


flop_mapping = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    quantized.conv2d: quant_conv_flop,
    quantized.conv2d_relu: quant_conv_flop,
    aten._scaled_dot_product_flash_attention: scaled_dot_product_flash_attention_flop,
}

unmapped_ops = set()


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class FlopCounterMode(TorchDispatchMode):
    def __init__(self, model=None):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ["Global"]
        # global mod
        if model is not None:
            for name, module in dict(model.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert self.parents[-1] == name
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)

        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        # print(f"Total: {sum(self.flop_counts['Global'].values()) / 1e9} GFLOPS")
        # for mod in self.flop_counts.keys():
        #     print(f"Module: ", mod)
        #     for k, v in self.flop_counts[mod].items():
        #         print(f"{k}: {v / 1e9} GFLOPS")
        #     print()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        else:
            unmapped_ops.add(func_packet)

        return out

    def get_flops(self):
        return sum(self.flop_counts["Global"].values()) / 1e9


def get_dims(module_name, height, width):
    # detection models have curated input sizes
    if module_name == "detection":
        # we can feed a batch of 1 for detection model instead of a list of 1 image
        dims = (3, height, width)
    elif module_name == "video":
        # hard-coding the time dimension to size 16
        dims = (1, 16, 3, height, width)
    else:
        dims = (1, 3, height, width)

    return dims


def get_ops(model: torch.nn.Module, weight: Weights, height=512, width=512):
    module_name = model.__module__.split(".")[-2]
    dims = get_dims(module_name=module_name, height=height, width=width)

    input_tensor = torch.randn(dims)

    # try:
    preprocess = weight.transforms()
    if module_name == "optical_flow":
        inp = preprocess(input_tensor, input_tensor)
    else:
        # hack to enable mod(*inp) for optical_flow models
        inp = [preprocess(input_tensor)]

    model.eval()

    flop_counter = FlopCounterMode(model)
    with flop_counter:
        # detection models expect a list of 3d tensors as inputs
        if module_name == "detection":
            model(inp)
        else:
            model(*inp)

        flops = flop_counter.get_flops()

    return round(flops, 3)


def get_file_size_mb(weight):
    weights_path = os.path.join(os.getenv("HOME"), ".cache/torch/hub/checkpoints", weight.url.split("/")[-1])
    weights_size_mb = os.path.getsize(weights_path) / 1024 / 1024

    return round(weights_size_mb, 3)
