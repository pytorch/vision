import copy
import operator
import torch
import warnings

from typing import Callable, Tuple


# TODO: create a util to undo the change
# TODO: if shouldn't have a _regularized_shotrcut when is has a downsample


class RegularizedShortcut(torch.nn.Module):
    def __init__(self, regularizer_layer: Callable[..., torch.nn.Module]):
        super().__init__()
        self._regularizer = regularizer_layer()

    def forward(self, input, result):
        return input + self._regularizer(result)


def add_regularized_shortcut(
        model: torch.nn.Module,
        block_types: Tuple[type, ...],
        regularizer_layer: Callable[..., torch.nn.Module],
        inplace: bool = True
) -> torch.nn.Module:
    if not inplace:
        model = copy.deepcopy(model)

    ATTR_NAME = "_regularized_shotrcut"
    tracer = torch.fx.Tracer()
    changed = False
    for m in model.modules():
        if isinstance(m, block_types):
            # Add the Layer directly on submodule prior tracing
            # workaround due to https://github.com/pytorch/pytorch/issues/66197
            m.add_module(ATTR_NAME, RegularizedShortcut(regularizer_layer))

            graph = tracer.trace(m)
            patterns = {operator.add, torch.add, "add"}

            input = None
            for node in graph.nodes:
                if node.op == 'call_function':
                    if node.target in patterns and len(node.args) == 2 and input in node.args:
                        with graph.inserting_after(node):
                            # Always put the shortcut value first
                            args = node.args if node.args[0] == input else node.args[::-1]
                            node.replace_all_uses_with(graph.call_module(ATTR_NAME, args))
                        graph.erase_node(node)
                        changed = True
                        break
                elif node.op == "placeholder":
                    input = node

            graph.lint()
    if not changed:
        warnings.warn("No shortcut was detected. Please ensure you have provided the correct `block_types` parameter "
                      "for this model.")

    return model


if __name__ == "__main__":
    from torchvision.models.resnet import resnet18, resnet50, BasicBlock, Bottleneck, load_state_dict_from_url
    from torchvision.ops.stochastic_depth import StochasticDepth
    from functools import partial

    regularizer_layer = partial(StochasticDepth, p=0.1, mode="row")
    model = resnet50()
    model = add_regularized_shortcut(model, (BasicBlock, Bottleneck), regularizer_layer)
    # print(model)
    out = model(torch.randn((7, 3, 224, 224)))
    print(out)

    # state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth")
    # model.load_state_dict(state_dict)

