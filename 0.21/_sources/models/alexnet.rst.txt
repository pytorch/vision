AlexNet
=======

.. currentmodule:: torchvision.models

The AlexNet model was originally introduced in the
`ImageNet Classification with Deep Convolutional Neural Networks
<https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__
paper. The implemented architecture is slightly different from the original one,
and is based on `One weird trick for parallelizing convolutional neural networks
<https://arxiv.org/abs/1404.5997>`__.


Model builders
--------------

The following model builders can be used to instantiate an AlexNet model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.alexnet.AlexNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    alexnet
