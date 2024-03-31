ConvNeXt
========

.. currentmodule:: torchvision.models

The ConvNeXt model is based on the `A ConvNet for the 2020s
<https://arxiv.org/abs/2201.03545>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a ConvNeXt model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.convnext.ConvNeXt`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_ for
more details about this class.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   convnext_tiny
   convnext_small
   convnext_base
   convnext_large
