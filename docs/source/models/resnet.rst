ResNet
======

.. currentmodule:: torchvision.models

The ResNet model is based on the `Deep Residual Learning for Image Recognition
<https://arxiv.org/abs/1512.03385>`_ paper.

.. note::
    Bottleneck in torchvision places the stride for downsampling at 3x3 convolution (``conv2``)
    while original implementation places the stride at the first 1x1 convolution (``conv1``)
    according to the paper.
    This variant improves the accuracy and it's known as `ResNet V1.5 
    <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

Model builders
--------------

The following model builders can be used to instantiate a ResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
