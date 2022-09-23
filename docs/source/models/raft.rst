RAFT
====

.. currentmodule:: torchvision.models.optical_flow

The RAFT model is based on the `RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow <https://arxiv.org/abs/2003.12039>`__ paper.


Model builders
--------------

The following model builders can be used to instantiate a RAFT model, with or
without pre-trained weights. All the model builders internally rely on the
``torchvision.models.optical_flow.RAFT`` base class. Please refer to the `source
code
<https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    raft_large
    raft_small
