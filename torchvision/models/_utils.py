from collections import OrderedDict

import torch
import torch.fx
from torch import nn
from typing import Dict, Any, Callable, Tuple, Optional


class IntermediateLayerGetter2(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# taken from https://github.com/pytorch/examples/blob/master/fx/module_tracer.py
# with slight modifications
class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            module_qualified_name = self.path_of_module(m)
            self.current_module_qualified_name = module_qualified_name
            if not self.is_leaf_module(m, module_qualified_name):
                out = forward(*args, **kwargs)
                self.node_to_originating_module[out.node] = module_qualified_name
                return out
            return self.create_proxy('call_module', module_qualified_name, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


def IntermediateLayerGetter(model: nn.Module, return_layers: Dict[str, str]) -> nn.Module:
    return_layers = {str(k): str(v) for k, v in return_layers.items()}

    # Instantiate our ModulePathTracer and use that to trace the model
    tracer = ModulePathTracer()
    graph = tracer.trace(model)

    name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    m = torch.fx.GraphModule(tracer.root, graph, name)

    # Get output node
    orig_output_node: Optional[torch.fx.Node] = None
    for n in reversed(m.graph.nodes):
        if n.op == "output":
            orig_output_node = n
            break
    assert orig_output_node
    # and remove it
    m.graph.erase_node(orig_output_node)

    # find output nodes corresponding to return_layers
    nodes = [n for n in m.graph.nodes]
    output_node = OrderedDict()
    for n in nodes:
        module_qualname = tracer.node_to_originating_module.get(n)
        if module_qualname in return_layers:
            output_node[return_layers[module_qualname]] = n

    # TODO raise error if some of return layers don't exist
    # TODO have duplicate nodes but full coverage for module names

    # and add them in the end of the graph
    with m.graph.inserting_after(nodes[-1]):
        m.graph.output(output_node)

    m.graph.eliminate_dead_code()
    m.recompile()

    # remove unused modules / parameters
    m = torch.fx.GraphModule(m, m.graph)
    return m
