MobileNet V2
============

.. currentmodule:: torchvision.models

.. warning::
    PyTorch Mobile is no longer actively supported. Please check out `ExecuTorch <https://pytorch.org/executorch-overview>`_, PyTorch’s all-new on-device inference library. You can also review our `end-to-end workflows <https://github.com/pytorch/executorch/tree/main/examples/portable#readme>`_ and the `source code <https://github.com/pytorch/executorch/tree/main/examples/models/mobilenet_v2>`_ for MobileNetV2.

The MobileNet V2 model is based on the `MobileNetV2: Inverted Residuals and Linear
Bottlenecks <https://arxiv.org/abs/1801.04381>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a MobileNetV2 model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.mobilenetv2.MobileNetV2`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mobilenet_v2
