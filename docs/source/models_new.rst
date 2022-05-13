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
keypoint detection, video classification and optical flow.

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

   models/alexnet
   models/convnext
   models/densenet
   models/efficientnet
   models/efficientnetv2
   models/googlenet
   models/inception
   models/mnasnet
   models/mobilenetv2
   models/mobilenetv3
   models/regnet
   models/resnet
   models/resnext
   models/shufflenetv2
   models/squeezenet
   models/swin_transformer
   models/vgg
   models/vision_transformer
   models/wide_resnet

|

Here is an example of how to use the pre-trained image classification models:

.. code:: python

    from torchvision.io import read_image
    from torchvision.models import resnet50, ResNet50_Weights

    img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)

    # Step 4: Use the model and print the predicted category
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score}%")


Table of all available classification weights
---------------------------------------------

Accuracies are reported on ImageNet

.. include:: generated/classification_table.rst

Quantized models
----------------

.. currentmodule:: torchvision.models.quantization

The following quantized classification models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/googlenet_quant
   models/mobilenetv2_quant

|

Here is an example of how to use the pre-trained quantized image classification models:

.. code:: python

    from torchvision.io import read_image
    from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

    img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model
    weights = ResNet50_QuantizedWeights.DEFAULT
    model = resnet50(weights=weights, quantize=True)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)

    # Step 4: Use the model and print the predicted category
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score}%")


Table of all available quantized classification weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accuracies are reported on ImageNet

.. include:: generated/classification_quant_table.rst

Semantic Segmentation
=====================

.. currentmodule:: torchvision.models.segmentation

The following semantic segmentation models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/deeplabv3
   models/fcn
   models/lraspp

Table of all available semantic segmentation weights
----------------------------------------------------

All models are evaluated on COCO val2017:

.. include:: generated/segmentation_table.rst



Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

Object Detection
----------------

.. currentmodule:: torchvision.models.detection

The following object detection models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/faster_rcnn
   models/fcos
   models/retinanet
   models/ssd
   models/ssdlite

Table of all available Object detection weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box MAPs are reported on COCO

.. include:: generated/detection_table.rst

Instance Segmentation
---------------------

.. currentmodule:: torchvision.models.detection

The following instance segmentation models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/mask_rcnn

Table of all available Instance segmentation weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box and Mask MAPs are reported on COCO

.. include:: generated/instance_segmentation_table.rst

Keypoint Detection
------------------

.. currentmodule:: torchvision.models.detection

The following keypoint detection models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/keypoint_rcnn

Table of all available Keypoint detection weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box and Keypoint MAPs are reported on COCO:

.. include:: generated/detection_keypoint_table.rst


Video Classification
====================

.. currentmodule:: torchvision.models.video

The following video classification models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/video_resnet

Table of all available video classification weights
---------------------------------------------------

Accuracies are reported on Kinetics-400

.. include:: generated/video_table.rst
