Feature extraction for model inspection
=======================================

.. currentmodule:: torchvision.models.feature_extraction

The ``torchvision.models.feature_extraction`` package contains
feature extraction utilities that let us tap into our models to access intermediate
transformations of our inputs. This could be useful for a variety of
applications in computer vision. Just a few examples are:

- Visualizing feature maps.
- Extracting features to compute image descriptors for tasks like facial
  recognition, copy-detection, or image retrieval.
- Passing selected features to downstream sub-networks for end-to-end training
  with a specific task in mind. For example, passing a hierarchy of features
  to a Feature Pyramid Network with object detection heads.

Torchvision provides :func:`create_feature_extractor` for this purpose.
It works by following roughly these steps:

1. Symbolically tracing the model to get a graphical representation of
   how it transforms the input, step by step.
2. Setting the user-selected graph nodes as outputs.
3. Removing all redundant nodes (anything downstream of the output nodes).
4. Generating python code from the resulting graph and bundling that into a
   PyTorch module together with the graph itself.

|

The `torch.fx documentation <https://pytorch.org/docs/stable/fx.html>`_
provides a more general and detailed explanation of the above procedure and
the inner workings of the symbolic tracing.

.. _about-node-names:

**About Node Names**

In order to specify which nodes should be output nodes for extracted
features, one should be familiar with the node naming convention used here
(which differs slightly from that used in ``torch.fx``). A node name is
specified as a ``.`` separated path walking the module hierarchy from top level
module down to leaf operation or leaf module. For instance ``"layer4.2.relu"``
in ResNet-50 represents the output of the ReLU of the 2nd block of the 4th
layer of the ``ResNet`` module. Here are some finer points to keep in mind:

- When specifying node names for :func:`create_feature_extractor`, you may
  provide a truncated version of a node name as a shortcut. To see how this
  works, try creating a ResNet-50 model and printing the node names with
  ``train_nodes, _ = get_graph_node_names(model) print(train_nodes)`` and
  observe that the last node pertaining to ``layer4`` is
  ``"layer4.2.relu_2"``. One may specify ``"layer4.2.relu_2"`` as the return
  node, or just ``"layer4"`` as this, by convention, refers to the last node
  (in order of execution) of ``layer4``.
- If a certain module or operation is repeated more than once, node names get
  an additional ``_{int}`` postfix to disambiguate. For instance, maybe the
  addition (``+``) operation is used three times in the same ``forward``
  method. Then there would be ``"path.to.module.add"``,
  ``"path.to.module.add_1"``, ``"path.to.module.add_2"``. The counter is
  maintained within the scope of the direct parent. So in ResNet-50 there is
  a ``"layer4.1.add"`` and a ``"layer4.2.add"``. Because the addition
  operations reside in different blocks, there is no need for a postfix to
  disambiguate.


**An Example**

Here is an example of how we might extract features for MaskRCNN:

.. code-block:: python

  import torch
  from torchvision.models import resnet50
  from torchvision.models.feature_extraction import get_graph_node_names
  from torchvision.models.feature_extraction import create_feature_extractor
  from torchvision.models.detection.mask_rcnn import MaskRCNN
  from torchvision.models.detection.backbone_utils import LastLevelMaxPool
  from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


  # To assist you in designing the feature extractor you may want to print out
  # the available nodes for resnet50.
  m = resnet50()
  train_nodes, eval_nodes = get_graph_node_names(resnet50())

  # The lists returned, are the names of all the graph nodes (in order of
  # execution) for the input model traced in train mode and in eval mode
  # respectively. You'll find that `train_nodes` and `eval_nodes` are the same
  # for this example. But if the model contains control flow that's dependent
  # on the training mode, they may be different.

  # To specify the nodes you want to extract, you could select the final node
  # that appears in each of the main layers:
  return_nodes = {
      # node_name: user-specified key for output dict
      'layer1.2.relu_2': 'layer1',
      'layer2.3.relu_2': 'layer2',
      'layer3.5.relu_2': 'layer3',
      'layer4.2.relu_2': 'layer4',
  }

  # But `create_feature_extractor` can also accept truncated node specifications
  # like "layer1", as it will just pick the last node that's a descendent of
  # of the specification. (Tip: be careful with this, especially when a layer
  # has multiple outputs. It's not always guaranteed that the last operation
  # performed is the one that corresponds to the output you desire. You should
  # consult the source code for the input model to confirm.)
  return_nodes = {
      'layer1': 'layer1',
      'layer2': 'layer2',
      'layer3': 'layer3',
      'layer4': 'layer4',
  }

  # Now you can build the feature extractor. This returns a module whose forward
  # method returns a dictionary like:
  # {
  #     'layer1': output of layer 1,
  #     'layer2': output of layer 2,
  #     'layer3': output of layer 3,
  #     'layer4': output of layer 4,
  # }
  create_feature_extractor(m, return_nodes=return_nodes)

  # Let's put all that together to wrap resnet50 with MaskRCNN

  # MaskRCNN requires a backbone with an attached FPN
  class Resnet50WithFPN(torch.nn.Module):
      def __init__(self):
          super(Resnet50WithFPN, self).__init__()
          # Get a resnet50 backbone
          m = resnet50()
          # Extract 4 main layers (note: MaskRCNN needs this particular name
          # mapping for return nodes)
          self.body = create_feature_extractor(
              m, return_nodes={f'layer{k}': str(v)
                               for v, k in enumerate([1, 2, 3, 4])})
          # Dry run to get number of channels for FPN
          inp = torch.randn(2, 3, 224, 224)
          with torch.no_grad():
              out = self.body(inp)
          in_channels_list = [o.shape[1] for o in out.values()]
          # Build FPN
          self.out_channels = 256
          self.fpn = FeaturePyramidNetwork(
              in_channels_list, out_channels=self.out_channels,
              extra_blocks=LastLevelMaxPool())

      def forward(self, x):
          x = self.body(x)
          x = self.fpn(x)
          return x


  # Now we can build our model!
  model = MaskRCNN(Resnet50WithFPN(), num_classes=91).eval()


API Reference
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    create_feature_extractor
    get_graph_node_names
