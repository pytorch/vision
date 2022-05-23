SwinTransformer
===============

.. currentmodule:: torchvision.models

The SwinTransformer model is based on the `Swin Transformer: Hierarchical Vision 
Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`__
paper.


Model builders
--------------

The following model builders can be used to instanciate an SwinTransformer model. 
`swin_t` can be instantiated with pre-trained weights and all others without. 
All the model builders internally rely on the ``torchvision.models.swin_transformer.SwinTransformer`` 
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    swin_t
    swin_s
    swin_b
