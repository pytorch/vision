DenseNet
========

.. currentmodule:: torchvision.models

The DenseNet model is based on the `Densely Connected Convolutional Networks
<https://arxiv.org/abs/1608.06993>`_ paper.


Model builders
--------------

The following model builders can be used to instantiate a DenseNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.densenet.DenseNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    densenet121
    densenet161
    densenet169
    densenet201
