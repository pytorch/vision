.. _models:

Models and pre-trained weights
##############################

The ``torchvision.models`` subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection, video classification, and optical flow.

General information on pre-trained weights
==========================================

TorchVision offers pre-trained weights for every provided architecture, using
the PyTorch :mod:`torch.hub`. Instancing a pre-trained model will download its
weights to a cache directory. This directory can be set using the `TORCH_HOME`
environment variable. See :func:`torch.hub.load_state_dict_from_url` for details.

.. note::

    The pre-trained models provided in this library may have their own licenses or
    terms and conditions derived from the dataset used for training. It is your
    responsibility to determine whether you have permission to use the models for
    your use case.

.. note ::
    Backward compatibility is guaranteed for loading a serialized
    ``state_dict`` to the model created using old PyTorch version.
    On the contrary, loading entire saved models or serialized
    ``ScriptModules`` (serialized using older versions of PyTorch)
    may not preserve the historic behaviour. Refer to the following
    `documentation
    <https://pytorch.org/docs/stable/notes/serialization.html#id6>`_


Initializing pre-trained models
-------------------------------

As of v0.13, TorchVision offers a new `Multi-weight support API
<https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/>`_
for loading different weights to the existing model builder methods:

.. code:: python

    from torchvision.models import resnet50, ResNet50_Weights

    # Old weights with accuracy 76.130%
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # New weights with accuracy 80.858%
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    resnet50(weights=ResNet50_Weights.DEFAULT)

    # Strings are also supported
    resnet50(weights="IMAGENET1K_V2")

    # No weights - random initialization
    resnet50(weights=None)


Migrating to the new API is very straightforward. The following method calls between the 2 APIs are all equivalent:

.. code:: python

    from torchvision.models import resnet50, ResNet50_Weights

    # Using pretrained weights:
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50(weights="IMAGENET1K_V1")
    resnet50(pretrained=True)  # deprecated
    resnet50(True)  # deprecated

    # Using no weights:
    resnet50(weights=None)
    resnet50()
    resnet50(pretrained=False)  # deprecated
    resnet50(False)  # deprecated

Note that the ``pretrained`` parameter is now deprecated, using it will emit warnings and will be removed on v0.15.

Using the pre-trained models
----------------------------

Before using the pre-trained models, one must preprocess the image
(resize with right resolution/interpolation, apply inference transforms,
rescale the values etc). There is no standard way to do this as it depends on
how a given model was trained. It can vary across model families, variants or
even weight versions. Using the correct preprocessing method is critical and
failing to do so may lead to decreased accuracy or incorrect outputs.

All the necessary information for the inference transforms of each pre-trained
model is provided on its weights documentation. To simplify inference, TorchVision
bundles the necessary preprocessing transforms into each model weight. These are
accessible via the ``weight.transforms`` attribute:

.. code:: python

    # Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    # Apply it to the input image
    img_transformed = preprocess(img)


Some models use modules which have different training and evaluation
behavior, such as batch normalization. To switch between these modes, use
``model.train()`` or ``model.eval()`` as appropriate. See
:meth:`~torch.nn.Module.train` or :meth:`~torch.nn.Module.eval` for details.

.. code:: python

    # Initialize model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Set model to eval mode
    model.eval()

Listing and retrieving available models
---------------------------------------

As of v0.14, TorchVision offers a new mechanism which allows listing and
retrieving models and weights by their names. Here are a few examples on how to
use them:

.. code:: python

    # List available models
    all_models = list_models()
    classification_models = list_models(module=torchvision.models)

    # Initialize models
    m1 = get_model("mobilenet_v3_large", weights=None)
    m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

    # Fetch weights
    weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
    assert weights == MobileNet_V3_Large_QuantizedWeights.DEFAULT

    weights_enum = get_model_weights("quantized_mobilenet_v3_large")
    assert weights_enum == MobileNet_V3_Large_QuantizedWeights

    weights_enum2 = get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
    assert weights_enum == weights_enum2

Here are the available public functions to retrieve models and their corresponding weights:

.. currentmodule:: torchvision.models
.. autosummary::
    :toctree: generated/
    :template: function.rst

    get_model
    get_model_weights
    get_weight
    list_models

Using models from Hub
---------------------

Most pre-trained models can be accessed directly via PyTorch Hub without having TorchVision installed:

.. code:: python

    import torch

    # Option 1: passing weights param as string
    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

    # Option 2: passing weights param as enum
    weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
    model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)

You can also retrieve all the available weights of a specific model via PyTorch Hub by doing:

.. code:: python

    import torch

    weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
    print([weight for weight in weight_enum])

The only exception to the above are the detection models included on
:mod:`torchvision.models.detection`. These models require TorchVision
to be installed because they depend on custom C++ operators.

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
   models/maxvit
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

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

The classes of the pre-trained model outputs can be found at ``weights.meta["categories"]``.

Table of all available classification weights
---------------------------------------------

Accuracies are reported on ImageNet-1K using single crops:

.. include:: generated/classification_table.rst

Quantized models
----------------

.. currentmodule:: torchvision.models.quantization

The following architectures provide support for INT8 quantized models, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/googlenet_quant
   models/inception_quant
   models/mobilenetv2_quant
   models/mobilenetv3_quant
   models/resnet_quant
   models/resnext_quant
   models/shufflenetv2_quant

|

Here is an example of how to use the pre-trained quantized image classification models:

.. code:: python

    from torchvision.io import read_image
    from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights

    img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_QuantizedWeights.DEFAULT
    model = resnet50(weights=weights, quantize=True)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score}%")

The classes of the pre-trained model outputs can be found at ``weights.meta["categories"]``.


Table of all available quantized classification weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accuracies are reported on ImageNet-1K using single crops:

.. include:: generated/classification_quant_table.rst

Semantic Segmentation
=====================

.. currentmodule:: torchvision.models.segmentation

.. betastatus:: segmentation module

The following semantic segmentation models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/deeplabv3
   models/fcn
   models/lraspp

|

Here is an example of how to use the pre-trained semantic segmentation models:

.. code:: python

    from torchvision.io.image import read_image
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    from torchvision.transforms.functional import to_pil_image

    img = read_image("gallery/assets/dog1.jpg")

    # Step 1: Initialize model with the best available weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["dog"]]
    to_pil_image(mask).show()

The classes of the pre-trained model outputs can be found at ``weights.meta["categories"]``.
The output format of the models is illustrated in :ref:`semantic_seg_output`.


Table of all available semantic segmentation weights
----------------------------------------------------

All models are evaluated a subset of COCO val2017, on the 20 categories that are present in the Pascal VOC dataset:

.. include:: generated/segmentation_table.rst


.. _object_det_inst_seg_pers_keypoint_det:

Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

The pre-trained models for detection, instance segmentation and
keypoint detection are initialized with the classification models
in torchvision. The models expect a list of ``Tensor[C, H, W]``.
Check the constructor of the models for more information.

.. betastatus:: detection module

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

|

Here is an example of how to use the pre-trained object detection models:

.. code:: python


    from torchvision.io.image import read_image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.utils import draw_bounding_boxes
    from torchvision.transforms.functional import to_pil_image

    img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()

The classes of the pre-trained model outputs can be found at ``weights.meta["categories"]``.
For details on how to plot the bounding boxes of the models, you may refer to :ref:`instance_seg_output`.

Table of all available Object detection weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box MAPs are reported on COCO val2017:

.. include:: generated/detection_table.rst


Instance Segmentation
---------------------

.. currentmodule:: torchvision.models.detection

The following instance segmentation models are available, with or without pre-trained
weights:

.. toctree::
   :maxdepth: 1

   models/mask_rcnn

|


For details on how to plot the masks of the models, you may refer to :ref:`instance_seg_output`.

Table of all available Instance segmentation weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box and Mask MAPs are reported on COCO val2017:

.. include:: generated/instance_segmentation_table.rst

Keypoint Detection
------------------

.. currentmodule:: torchvision.models.detection

The following person keypoint detection models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/keypoint_rcnn

|

The classes of the pre-trained model outputs can be found at ``weights.meta["keypoint_names"]``.
For details on how to plot the bounding boxes of the models, you may refer to :ref:`keypoint_output`.

Table of all available Keypoint detection weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Box and Keypoint MAPs are reported on COCO val2017:

.. include:: generated/detection_keypoint_table.rst


Video Classification
====================

.. currentmodule:: torchvision.models.video

.. betastatus:: video module

The following video classification models are available, with or without
pre-trained weights:

.. toctree::
   :maxdepth: 1

   models/video_mvit
   models/video_resnet
   models/video_s3d
   models/video_swin_transformer

|

Here is an example of how to use the pre-trained video classification models:

.. code:: python


    from torchvision.io.video import read_video
    from torchvision.models.video import r3d_18, R3D_18_Weights

    vid, _, _ = read_video("test/assets/videos/v_SoccerJuggling_g23_c01.avi", output_format="TCHW")
    vid = vid[:32]  # optionally shorten duration

    # Step 1: Initialize model with the best available weights
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(vid).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    label = prediction.argmax().item()
    score = prediction[label].item()
    category_name = weights.meta["categories"][label]
    print(f"{category_name}: {100 * score}%")

The classes of the pre-trained model outputs can be found at ``weights.meta["categories"]``.


Table of all available video classification weights
---------------------------------------------------

Accuracies are reported on Kinetics-400 using single crops for clip length 16:

.. include:: generated/video_table.rst

Optical Flow
============

.. currentmodule:: torchvision.models.optical_flow

The following Optical Flow models are available, with or without pre-trained

.. toctree::
   :maxdepth: 1

   models/raft
