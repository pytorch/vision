SwinTransformer
===============

.. currentmodule:: torchvision.models

The SwinTransformer models are based on the `Swin Transformer: Hierarchical Vision
Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`__
paper.
SwinTransformer V2 models are based on the `Swin Transformer V2: Scaling Up Capacity
and Resolution <https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.pdf>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate an SwinTransformer model (original and V2) with and without pre-trained weights.
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
    swin_v2_t
    swin_v2_s
    swin_v2_b
