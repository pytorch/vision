from typing import Any, Dict, Callable, List, Union
from collections import OrderedDict
import warnings
import re
from pprint import pprint
from inspect import ismethod

import torch
from torch import Tensor
from torch import nn
from torch import fx


class IntermediateLayerGetter(nn.ModuleDict):
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

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class NodePathTracer(fx.Tracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    qualified name of the Node from which the operation originated. A
    qualified name here is a `.` seperated path walking the hierarchy from top
    level module down to leaf operation or leaf module. The name of the top
    level module is not included as part of the qualified name. For example,
    if we trace a module who's forward method applies a ReLU module, the
    qualified name for that node will simply be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its qualified name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate qualified name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """
    def __init__(self, *args, **kwargs):
        super(NodePathTracer, self).__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ''
        # A map from FX Node to the qualified name
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy('call_module', module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(self, kind: str, target: fx.node.Target, args, kwargs,
                     name=None, type_expr=None) -> fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(
            self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(
            self, module_qualname: str, node: fx.node.Node) -> str:
        node_qualname = module_qualname
        if node.op == 'call_module':
            # Node terminates in a leaf module so the module_qualname is a
            # complete description of the node
            for existing_qualname in reversed(self.node_to_qualname.values()):
                # Check to see if existing_qualname is of the form
                # {node_qualname} or {node_qualname}_{int}
                if re.match(rf'{node_qualname}(_[0-9]+)?$',
                            existing_qualname) is not None:
                    postfix = existing_qualname.replace(node_qualname, '')
                    if len(postfix):
                        # Existing_qualname is of the form {node_qualname}_{int}
                        next_index = int(postfix[1:]) + 1
                    else:
                        # existing_qualname is of the form {node_qualname}
                        next_index = 1
                    node_qualname += f'_{next_index}'
                    break
        else:
            # Node terminates in non- leaf module so the node name needs to be
            # appended
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += '.'
            node_qualname += str(node)
        return node_qualname


def print_graph_node_qualified_names(
        model: nn.Module, tracer_kwargs: Dict = {}):
    """
    Dev utility to prints nodes in order of execution. Useful for choosing
    nodes for a FeatureGraphNet design. There are two reasons that qualified
    node names can't easily be read directly from the code for a model:
        1. Not all submodules are traced through. Modules from `torch.nn` all
           fall within this category.
        2. Node qualified names that occur more than once in the graph get a
           `_{counter}` postfix.

    Args:
        model (nn.Module): model on which we will extract the features
        tracer_kwargs (Dict): a dictionary of keywork arguments for
            `NodePathTracer` (which passes them onto it's parent class
            `torch.fx.Tracer`).
    """
    tracer = NodePathTracer(**tracer_kwargs)
    tracer.trace(model)
    pprint(list(tracer.node_to_qualname.values()))


def build_feature_graph_net(
        model: nn.Module,
        return_nodes: Union[List[str], Dict[str, str]],
        tracer_kwargs: Dict = {}) -> fx.GraphModule:
    """
    Creates a new graph module that returns intermediate nodes from a given
    model as dictionary with user specified keys as strings, and the requested
    outputs as values. This is achieved by re-writing the computation graph of
    the model via FX to return the desired nodes as outputs. All unused nodes
    are removed, together with their corresponding parameters.

    A note on node specification: A node qualified name is specified as a `.`
    seperated path walking the hierarchy from top level module down to leaf
    operation or leaf module. For instance `blocks.5.3.bn1`. The keys of the
    `return_nodes` argument should point to either a node's qualified name,
    or some truncated version of it. For example, one could provide `blocks.5`
    as a key, and the last node with that prefix will be selected.
    `print_graph_node_qualified_names` is a useful helper function for getting
    a list of qualified names of a model.

    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (Union[List[name], Dict[name, new_name]])): either a list
            or a dict containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a `Dict`, the keys are the qualified node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a `List`, it is treated as a `Dict` mapping
            node specification strings directly to output names.
        tracer_kwargs (Dict): a dictionary of keywork arguments for
            `NodePathTracer` (which passes them onto it's parent class
            `torch.fx.Tracer`).

    NOTE: Static control flow will be frozen into place for the resulting
        `GraphModule`. Among other consequences, this means that control flow
        that relies on whether the model is in train or eval mode will be
        frozen into place (except for leaf modules which are not traced
        through). Therefore, calling `.train()` or `.eval()` on the resulting
        `GraphModule` may not have all the desired effects.

    Examples::

        >>> model = torchvision.models.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> graph_module = torchvision.models._utils.build_feature_graph_net(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = graph_module(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

    """
    if isinstance(return_nodes, list):
        return_nodes = {n: n for n in return_nodes}
    return_nodes = {str(k): str(v) for k, v in return_nodes.items()}

    # Instantiate our NodePathTracer and use that to trace the model
    tracer = NodePathTracer(**tracer_kwargs)
    graph = tracer.trace(model)

    name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    graph_module = fx.GraphModule(tracer.root, graph, name)

    available_nodes = [f'{v}.{k}' for k, v in tracer.node_to_qualname.items()]
    # FIXME We don't know if we should expect this to happen
    assert len(set(available_nodes)) == len(available_nodes), \
        "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
    # Check that all outputs in return_nodes are present in the model
    for query in return_nodes.keys():
        if not any([m.startswith(query) for m in available_nodes]):
            raise ValueError(f"return_node: {query} is not present in model")

    # Remove existing output nodes
    orig_output_node = None
    for n in reversed(graph_module.graph.nodes):
        if n.op == "output":
            orig_output_node = n
    assert orig_output_node
    # And remove it
    graph_module.graph.erase_node(orig_output_node)
    # Find nodes corresponding to return_nodes and make them into output_nodes
    nodes = [n for n in graph_module.graph.nodes]
    output_nodes = OrderedDict()
    for n in reversed(nodes):
        if 'tensor_constant' in str(n):
            # NOTE Without this control flow we would get a None value for
            # `module_qualname = tracer.node_to_qualname.get(n)`.
            # On the other hand, we can safely assume that we'll never need to
            # get this as an interesting intermediate node.
            continue
        module_qualname = tracer.node_to_qualname.get(n)
        for query in return_nodes:
            depth = query.count('.')
            if '.'.join(module_qualname.split('.')[:depth + 1]) == query:
                output_nodes[return_nodes[query]] = n
                return_nodes.pop(query)
                break
    output_nodes = OrderedDict(reversed(list(output_nodes.items())))

    # And add them in the end of the graph
    with graph_module.graph.inserting_after(nodes[-1]):
        graph_module.graph.output(output_nodes)

    # Remove unused modules / parameters
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()
    graph_module = fx.GraphModule(graph_module, graph_module.graph, name)
    return graph_module


class FeatureGraphNet(nn.Module):
    """
    Wrap a `GraphModule` from `build_feature_graph_net` while also keeping the
    original model's non-parameter properties for reference. The original
    model's paremeters are discarded.

    See `build_feature_graph_net` docstring for more  information.

    NOTE: This puts the input model into eval mode prior to tracing. This
        means that any control flow dependent on the model being in train mode
        will be lost.
    """
    def __init__(self, model: nn.Module,
                 return_nodes: Union[List[str], Dict[str, str]],
                 tracer_kwargs: Dict = {}):
        """
        Args:
            model (nn.Module): model on which we will extract the features
            return_nodes (Union[List[name], Dict[name, new_name]])): either a list
                or a dict containing the names (or partial names - see note above)
                of the nodes for which the activations will be returned. If it is
                a `Dict`, the keys are the qualified node names, and the values
                are the user-specified keys for the graph module's returned
                dictionary. If it is a `List`, it is treated as a `Dict` mapping
                node specification strings directly to output names.
            tracer_kwargs (Dict): a dictionary of keywork arguments for
                `NodePathTracer` (which passes them onto it's parent class
                `torch.fx.Tracer`).
        """
        super(FeatureGraphNet, self).__init__()
        model.eval()
        self.graph_module = build_feature_graph_net(
            model, return_nodes, tracer_kwargs)
        # Keep non-parameter model properties for reference
        for attr_str in model.__dir__():
            attr = getattr(model, attr_str)
            if (not attr_str.startswith('_')
                    and attr_str not in self.__dir__()
                    and not ismethod(attr)
                    and not isinstance(attr, (nn.Module, nn.Parameter))):
                setattr(self, attr_str, attr)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.graph_module(x)

    def train(self, mode: bool = True):
        """
        NOTE: This also covers `self.eval()` as that just calls self.train(False)
        """
        if mode:
            warnings.warn(
                "Setting a FeatureGraphNet to training mode won't necessarily"
                " have the desired effect. Control flow depending on"
                " `self.training` will follow the `False` path. See"
                " `FeatureGraphNet` doc-string for more details.")

        super(FeatureGraphNet, self).train(mode)
