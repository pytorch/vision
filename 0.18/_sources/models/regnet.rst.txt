RegNet
======

.. currentmodule:: torchvision.models

The RegNet model is based on the `Designing Network Design Spaces
<https://arxiv.org/abs/2003.13678>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a RegNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.regnet.RegNet`` base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_ for
more details about this class.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   regnet_y_400mf
   regnet_y_800mf
   regnet_y_1_6gf
   regnet_y_3_2gf
   regnet_y_8gf
   regnet_y_16gf
   regnet_y_32gf
   regnet_y_128gf
   regnet_x_400mf
   regnet_x_800mf
   regnet_x_1_6gf
   regnet_x_3_2gf
   regnet_x_8gf
   regnet_x_16gf
   regnet_x_32gf
