torchvision.feature_extraction
==============================

.. currentmodule:: torchvision.models.feature_extraction

Feature extraction utilities let us tap into our models to access intermediate
transformations of our inputs. This could be useful for a variety of
applications in computer vision. Just a few examples are:

- Visualizing feature maps.
- Extracting features to compute image descriptors for tasks like facial
  recognition, copy-detection, or image retrieval.
- Passing selected features to downstream sub-networks for end-to-end training
  with a specific task in mind. For example, passing a hierarchy of features
  to a Feature Pyramid Network with object detection heads.

Torchvision provides two helpers for doing feature extraction. :class:`IntermediateLayerGetter`
is easy to understand and effective. That said, it only allows coarse control
over which features are extracted, and makes some assumptions about the layout
of the input module. :func:`build_feature_graph_net` is far more
flexible, but does have some rough edges as it requires that the input model
is symbolically traceable (see 
`torch.fx documentation <https://pytorch.org/docs/stable/fx.html>`_ for more
information on symbolic tracing).

.. autoclass:: IntermediateLayerGetter

.. autofunction:: build_feature_graph_net

.. autofunction:: print_graph_node_qualified_names