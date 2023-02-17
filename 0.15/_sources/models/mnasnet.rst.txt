MNASNet
=======

.. currentmodule:: torchvision.models


The MNASNet model is based on the `MnasNet: Platform-Aware Neural Architecture
Search for Mobile <https://arxiv.org/pdf/1807.11626.pdf>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate an MNASNet model.
All the model builders internally rely on the
``torchvision.models.mnasnet.MNASNet`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mnasnet0_5
    mnasnet0_75
    mnasnet1_0
    mnasnet1_3
