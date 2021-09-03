torchvision.models.feature_extraction
=====================================

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

Torchvision provides :func:`build_feature_extractor` for this purpose.
It works by following roughly these steps:

1. Symbolically tracing the model to get a graphical representation of
   how it transforms the input, step by step.
2. Setting the user-selected graph nodes as ouputs.
3. Removing all redundant nodes (anything downstream of the ouput nodes).
4. Generating python code from the resulting graph and bundling that, together
   with the graph, and bundling that into a PyTorch module.

|

See `torch.fx documentation <https://pytorch.org/docs/stable/fx.html>`_ for
more information on symbolic tracing.

.. autofunction:: build_feature_extractor

.. autofunction:: get_graph_node_names