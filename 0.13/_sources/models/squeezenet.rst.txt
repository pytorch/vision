SqueezeNet
==========

.. currentmodule:: torchvision.models

The SqueezeNet model is based on the `SqueezeNet: AlexNet-level accuracy with
50x fewer parameters and <0.5MB model size <https://arxiv.org/abs/1602.07360>`__
paper.


Model builders
--------------

The following model builders can be used to instantiate a SqueezeNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.squeezenet.SqueezeNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    squeezenet1_0
    squeezenet1_1
