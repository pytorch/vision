MaxVit
===============

.. currentmodule:: torchvision.models

The MaxVit transformer models are based on the `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate an MaxVit model with and without pre-trained weights.
All the model builders internally rely on the ``torchvision.models.maxvit.MaxVit`` 
base class. Please refer to the `source code
<https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    maxvit_t
