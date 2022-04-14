.. _models:

Models and pre-trained weights
##############################


The ``torchvision.models`` subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection, video classification, and optical flow.

.. note ::
    Backward compatibility is guaranteed for loading a serialized 
    ``state_dict`` to the model created using old PyTorch version. 
    On the contrary, loading entire saved models or serialized 
    ``ScriptModules`` (seralized using older versions of PyTorch) 
    may not preserve the historic behaviour. Refer to the following 
    `documentation 
    <https://pytorch.org/docs/stable/notes/serialization.html#id6>`_   


Classification
==============

.. currentmodule:: torchvision.models

The following classification models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/resnet
   models/vgg


Table of all available classificaiton weights
---------------------------------------------

Accuracies are reported on ImageNet

.. include:: generated/classification_table.rst


Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

TODO: Something similar to classification models: list of models + table of weights
