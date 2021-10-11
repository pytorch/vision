import copy
import operator
import warnings
from typing import Callable, Optional, Tuple, Union

import torch
from torch import fx
from torchvision.models.feature_extraction import LeafModuleAwareTracer


# TODO: Investigate what happens in the scenario of y = x + f1(x) + f2(x).


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

    reg_name = RegularizedShortcut.__name__.lower()
    tracer = fx.Tracer()
    modifications = {}
    for name, m in model.named_modules():
        if isinstance(m, block_types):
            # Add the Layer directly on submodule prior tracing
            # workaround due to https://github.com/pytorch/pytorch/issues/66197
            m.add_module(reg_name, RegularizedShortcut(regularizer_layer))

            graph = tracer.trace(m)
            patterns = {operator.add, torch.add, "add"}

            input = None
            for node in graph.nodes:
                if node.op == "call_function":
                    if node.target in patterns and len(node.args) == 2 and input in node.args:
                        # TODO: ensure the arg2 has "input" as its ancestor
                        with graph.inserting_after(node):
                            # Always put the shortcut value first
                            args = node.args if node.args[0] == input else node.args[::-1]
                            node.replace_all_uses_with(graph.call_module(reg_name, args))
                        graph.erase_node(node)
                        modifications[name] = graph
                        break
                elif node.op == "placeholder":
                    input = node

    if modifications:
        # Update the model by overwriting its modules
        for name, graph in modifications.items():
            graph.lint()
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            previous_child = parent.get_submodule(child_name)
            new_child = fx.GraphModule(previous_child, graph, previous_child.__class__.__name__)
            parent.register_module(child_name, new_child)
    else:
        warnings.warn(
            "No shortcut was detected. Please ensure you have provided the correct `block_types` parameter "
            "for this model."
        )

    return model


def del_regularized_shortcut(
    model: torch.nn.Module,
    block_types: Union[type, Tuple[type, ...]] = RegularizedShortcut,
    op: Optional[Callable] = operator.add,
    inplace: bool = True,
) -> torch.nn.Module:
    if isinstance(block_types, type):
        block_types = (block_types,)
    if not inplace:
        model = copy.deepcopy(model)

    tracer = LeafModuleAwareTracer(leaf_modules=block_types)
    graph = tracer.trace(model)
    for node in graph.nodes:
        # The isinstance() won't work if the model has already been traced before because it loses
        # the class info of submodules. See https://github.com/pytorch/pytorch/issues/66335
        if node.op == "call_module" and isinstance(model.get_submodule(node.target), block_types):
            if op is not None:
                with graph.inserting_before(node):
                    new_node = graph.call_function(op, node.args)
                    node.replace_all_uses_with(new_node)
            else:
                if len(node.args) == 1:
                    node.replace_all_uses_with(node.prev)
                else:
                    raise ValueError("Can't eliminate an operator that receives more than 1 arguments.")
            graph.erase_node(node)

    return fx.GraphModule(model, graph, model.__class__.__name__)


if __name__ == "__main__":
    from functools import partial

    from torchvision.models.resnet import resnet50, BasicBlock, Bottleneck
    from torchvision.ops.stochastic_depth import StochasticDepth

    out = []
    batch = torch.randn((7, 3, 224, 224))

    print("Before")
    model = resnet50()
    with torch.no_grad():
        out.append(model(batch))
    fx.symbolic_trace(model).graph.print_tabular()

    print("After addition")
    regularizer_layer = partial(StochasticDepth, p=0.0, mode="row")
    model = add_regularized_shortcut(model, (BasicBlock, Bottleneck), regularizer_layer)
    fx.symbolic_trace(model).graph.print_tabular()
    # print(model)
    with torch.no_grad():
        out.append(model(batch))

    print("After deletion")
    model = del_regularized_shortcut(model)
    fx.symbolic_trace(model).graph.print_tabular()
    with torch.no_grad():
        out.append(model(batch))

    for v in out[1:]:
        torch.testing.assert_allclose(out[0], v)
