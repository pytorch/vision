import copy
import operator
import warnings
from typing import Callable, Tuple, Union

import torch
from torch import fx
from torchvision.models.feature_extraction import LeafModuleAwareTracer


_MODULE_NAME = "_regularized_shotrcut"


class RegularizedShortcut(torch.nn.Module):
    def __init__(self, regularizer_layer: Callable[..., torch.nn.Module]):
        super().__init__()
        self._regularizer = regularizer_layer()

    def forward(self, input, result):
        return input + self._regularizer(result)


def add_regularized_shortcut(
    model: torch.nn.Module,
    block_types: Union[type, Tuple[type, ...]],
    regularizer_layer: Callable[..., torch.nn.Module],
    inplace: bool = True,
) -> torch.nn.Module:
    if not inplace:
        model = copy.deepcopy(model)

    tracer = fx.Tracer()
    modifications = {}
    for name, m in model.named_modules():
        if isinstance(m, block_types):
            # Add the Layer directly on submodule prior tracing
            # workaround due to https://github.com/pytorch/pytorch/issues/66197
            m.add_module(_MODULE_NAME, RegularizedShortcut(regularizer_layer))

            graph = tracer.trace(m)
            patterns = {operator.add, torch.add, "add"}

            input = None
            for node in graph.nodes:
                if node.op == "call_function":
                    if node.target in patterns and len(node.args) == 2 and input in node.args:
                        with graph.inserting_after(node):
                            # Always put the shortcut value first
                            args = node.args if node.args[0] == input else node.args[::-1]
                            node.replace_all_uses_with(graph.call_module(_MODULE_NAME, args))
                        graph.erase_node(node)
                        graph.lint()
                        modifications[name] = graph
                elif node.op == "placeholder":
                    input = node

    if modifications:
        # Update the model by overwriting its modules
        for name, graph in modifications.items():
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            previous_child = parent.get_submodule(child_name)
            new_child = fx.GraphModule(previous_child, graph)
            parent.register_module(child_name, new_child)
    else:
        warnings.warn(
            "No shortcut was detected. Please ensure you have provided the correct `block_types` parameter "
            "for this model."
        )

    return model


def del_regularized_shortcut(model: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    if not inplace:
        model = copy.deepcopy(model)

    tracer = LeafModuleAwareTracer(leaf_modules=[RegularizedShortcut])
    graph = tracer.trace(model)
    for node in graph.nodes:
        if node.op == "call_module" and node.target.rsplit(".", 1)[-1] == _MODULE_NAME:
            with graph.inserting_before(node):
                new_node = graph.call_function(operator.add, node.args)
                node.replace_all_uses_with(new_node)
            graph.erase_node(node)

    return fx.GraphModule(model, graph)


if __name__ == "__main__":
    from functools import partial

    from torchvision.models.resnet import resnet50, BasicBlock, Bottleneck
    from torchvision.ops.stochastic_depth import StochasticDepth

    out = []
    batch = torch.randn((7, 3, 224, 224))

    print("Before")
    model = resnet50()
    out.append(model(batch))
    fx.symbolic_trace(model).graph.print_tabular()

    print("After addition")
    regularizer_layer = partial(StochasticDepth, p=0.0, mode="row")
    model = add_regularized_shortcut(model, (BasicBlock, Bottleneck), regularizer_layer)
    fx.symbolic_trace(model).graph.print_tabular()
    # print(model)
    out.append(model(batch))
    # state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth")
    # model.load_state_dict(state_dict)

    print("After deletion")
    model = del_regularized_shortcut(model)
    fx.symbolic_trace(model).graph.print_tabular()
    out.append(model(batch))

    for v in out[1:]:
        torch.testing.assert_allclose(out[0], v)
