VisionTransformer
=================

.. currentmodule:: torchvision.models

The VisionTransformer model is based on the `An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a VisionTransformer model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.vision_transformer.VisionTransformer`` base class.
Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_ for
more details about this class.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   vit_b_16
   vit_b_32
   vit_l_16
   vit_l_32
   vit_h_14
