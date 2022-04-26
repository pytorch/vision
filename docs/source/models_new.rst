.. _models_new:

Models and pre-trained weights - New
####################################

.. note::

    These are the new models docs, documenting the new multi-weight API.
    TODO: Once all is done, remove the "- New" part in the title above, and
    rename this file as models.rst


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

   models/convnext
   models/densenet
   models/efficientnet
   models/efficientnetv2
   models/googlenet
   models/regnet
   models/resnet
   models/resnext
   models/squeezenet
   models/vgg
   models/vision_transformer


Table of all available classification weights
---------------------------------------------

Accuracies are reported on ImageNet

.. include:: generated/classification_table.rst


Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

.. currentmodule:: torchvision.models.detection

The following detection models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/retinanet

Table of all available detection weights
----------------------------------------

Box MAPs are reported on COCO

.. include:: generated/detection_table.rst
