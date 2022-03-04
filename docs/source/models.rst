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

The models subpackage contains definitions for the following model
architectures for image classification:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `SqueezeNet`_
-  `DenseNet`_
-  `Inception`_ v3
-  `GoogLeNet`_
-  `ShuffleNet`_ v2
-  `MobileNetV2`_
-  `MobileNetV3`_
-  `ResNeXt`_
-  `Wide ResNet`_
-  `MNASNet`_
-  `EfficientNet`_ v1 & v2
-  `RegNet`_
-  `VisionTransformer`_
-  `ConvNeXt`_

You can construct a model with random weights by calling its constructor:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet161()
    inception = models.inception_v3()
    googlenet = models.googlenet()
    shufflenet = models.shufflenet_v2_x1_0()
    mobilenet_v2 = models.mobilenet_v2()
    mobilenet_v3_large = models.mobilenet_v3_large()
    mobilenet_v3_small = models.mobilenet_v3_small()
    resnext50_32x4d = models.resnext50_32x4d()
    wide_resnet50_2 = models.wide_resnet50_2()
    mnasnet = models.mnasnet1_0()
    efficientnet_b0 = models.efficientnet_b0()
    efficientnet_b1 = models.efficientnet_b1()
    efficientnet_b2 = models.efficientnet_b2()
    efficientnet_b3 = models.efficientnet_b3()
    efficientnet_b4 = models.efficientnet_b4()
    efficientnet_b5 = models.efficientnet_b5()
    efficientnet_b6 = models.efficientnet_b6()
    efficientnet_b7 = models.efficientnet_b7()
    efficientnet_v2_s = models.efficientnet_v2_s()
    efficientnet_v2_m = models.efficientnet_v2_m()
    efficientnet_v2_l = models.efficientnet_v2_l()
    regnet_y_400mf = models.regnet_y_400mf()
    regnet_y_800mf = models.regnet_y_800mf()
    regnet_y_1_6gf = models.regnet_y_1_6gf()
    regnet_y_3_2gf = models.regnet_y_3_2gf()
    regnet_y_8gf = models.regnet_y_8gf()
    regnet_y_16gf = models.regnet_y_16gf()
    regnet_y_32gf = models.regnet_y_32gf()
    regnet_y_128gf = models.regnet_y_128gf()
    regnet_x_400mf = models.regnet_x_400mf()
    regnet_x_800mf = models.regnet_x_800mf()
    regnet_x_1_6gf = models.regnet_x_1_6gf()
    regnet_x_3_2gf = models.regnet_x_3_2gf()
    regnet_x_8gf = models.regnet_x_8gf()
    regnet_x_16gf = models.regnet_x_16gf()
    regnet_x_32gf = models.regnet_x_32gf()
    vit_b_16 = models.vit_b_16()
    vit_b_32 = models.vit_b_32()
    vit_l_16 = models.vit_l_16()
    vit_l_32 = models.vit_l_32()
    convnext_tiny = models.convnext_tiny()
    convnext_small = models.convnext_small()
    convnext_base = models.convnext_base()
    convnext_large = models.convnext_large()

We provide pre-trained models, using the PyTorch :mod:`torch.utils.model_zoo`.
These can be constructed by passing ``pretrained=True``:

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)
    googlenet = models.googlenet(pretrained=True)
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    mnasnet = models.mnasnet1_0(pretrained=True)
    efficientnet_b0 = models.efficientnet_b0(pretrained=True)
    efficientnet_b1 = models.efficientnet_b1(pretrained=True)
    efficientnet_b2 = models.efficientnet_b2(pretrained=True)
    efficientnet_b3 = models.efficientnet_b3(pretrained=True)
    efficientnet_b4 = models.efficientnet_b4(pretrained=True)
    efficientnet_b5 = models.efficientnet_b5(pretrained=True)
    efficientnet_b6 = models.efficientnet_b6(pretrained=True)
    efficientnet_b7 = models.efficientnet_b7(pretrained=True)
    efficientnet_v2_s = models.efficientnet_v2_s(pretrained=True)
    efficientnet_v2_m = models.efficientnet_v2_m(pretrained=True)
    efficientnet_v2_l = models.efficientnet_v2_l(pretrained=True)
    regnet_y_400mf = models.regnet_y_400mf(pretrained=True)
    regnet_y_800mf = models.regnet_y_800mf(pretrained=True)
    regnet_y_1_6gf = models.regnet_y_1_6gf(pretrained=True)
    regnet_y_3_2gf = models.regnet_y_3_2gf(pretrained=True)
    regnet_y_8gf = models.regnet_y_8gf(pretrained=True)
    regnet_y_16gf = models.regnet_y_16gf(pretrained=True)
    regnet_y_32gf = models.regnet_y_32gf(pretrained=True)
    regnet_x_400mf = models.regnet_x_400mf(pretrained=True)
    regnet_x_800mf = models.regnet_x_800mf(pretrained=True)
    regnet_x_1_6gf = models.regnet_x_1_6gf(pretrained=True)
    regnet_x_3_2gf = models.regnet_x_3_2gf(pretrained=True)
    regnet_x_8gf = models.regnet_x_8gf(pretrained=True)
    regnet_x_16gf = models.regnet_x_16gf(pretrainedTrue)
    regnet_x_32gf = models.regnet_x_32gf(pretrained=True)
    vit_b_16 = models.vit_b_16(pretrained=True)
    vit_b_32 = models.vit_b_32(pretrained=True)
    vit_l_16 = models.vit_l_16(pretrained=True)
    vit_l_32 = models.vit_l_32(pretrained=True)
    convnext_tiny = models.convnext_tiny(pretrained=True)
    convnext_small = models.convnext_small(pretrained=True)
    convnext_base = models.convnext_base(pretrained=True)
    convnext_large = models.convnext_large(pretrained=True)

Instancing a pre-trained model will download its weights to a cache directory.
This directory can be set using the `TORCH_HOME` environment variable. See
:func:`torch.hub.load_state_dict_from_url` for details.

Some models use modules which have different training and evaluation
behavior, such as batch normalization. To switch between these modes, use
``model.train()`` or ``model.eval()`` as appropriate. See
:meth:`~torch.nn.Module.train` or :meth:`~torch.nn.Module.eval` for details.

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
where H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
You can use the following transform to normalize::

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

An example of such normalization can be found in the imagenet example
`here <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>`_

The process for obtaining the values of `mean` and `std` is roughly equivalent
to::

    import torch
    from torchvision import datasets, transforms as T

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor(), T.ConvertImageDtype(torch.float)])
    dataset = datasets.ImageNet(".", split="train", transform=transform)

    means = []
    stds = []
    for img in subset(dataset):
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

Unfortunately, the concrete `subset` that was used is lost. For more
information see `this discussion <https://github.com/pytorch/vision/issues/1439>`_
or `these experiments <https://github.com/pytorch/vision/pull/1965>`_.

The sizes of the EfficientNet models depend on the variant. For the exact input sizes
`check here <https://github.com/pytorch/vision/blob/d2bfd639e46e1c5dc3c177f889dc7750c8d137c7/references/classification/train.py#L92-L93>`_

ImageNet 1-crop error rates

================================  =============   =============
Model                             Acc@1           Acc@5
================================  =============   =============
AlexNet                           56.522          79.066
VGG-11                            69.020          88.628
VGG-13                            69.928          89.246
VGG-16                            71.592          90.382
VGG-19                            72.376          90.876
VGG-11 with batch normalization   70.370          89.810
VGG-13 with batch normalization   71.586          90.374
VGG-16 with batch normalization   73.360          91.516
VGG-19 with batch normalization   74.218          91.842
ResNet-18                         69.758          89.078
ResNet-34                         73.314          91.420
ResNet-50                         76.130          92.862
ResNet-101                        77.374          93.546
ResNet-152                        78.312          94.046
SqueezeNet 1.0                    58.092          80.420
SqueezeNet 1.1                    58.178          80.624
Densenet-121                      74.434          91.972
Densenet-169                      75.600          92.806
Densenet-201                      76.896          93.370
Densenet-161                      77.138          93.560
Inception v3                      77.294          93.450
GoogleNet                         69.778          89.530
ShuffleNet V2 x1.0                69.362          88.316
ShuffleNet V2 x0.5                60.552          81.746
MobileNet V2                      71.878          90.286
MobileNet V3 Large                74.042          91.340
MobileNet V3 Small                67.668          87.402
ResNeXt-50-32x4d                  77.618          93.698
ResNeXt-101-32x8d                 79.312          94.526
Wide ResNet-50-2                  78.468          94.086
Wide ResNet-101-2                 78.848          94.284
MNASNet 1.0                       73.456          91.510
MNASNet 0.5                       67.734          87.490
EfficientNet-B0                   77.692          93.532
EfficientNet-B1                   78.642          94.186
EfficientNet-B2                   80.608          95.310
EfficientNet-B3                   82.008          96.054
EfficientNet-B4                   83.384          96.594
EfficientNet-B5                   83.444          96.628
EfficientNet-B6                   84.008          96.916
EfficientNet-B7                   84.122          96.908
EfficientNetV2-s                  84.228          96.878
EfficientNetV2-m                  85.112          97.156
EfficientNetV2-l                  85.810          97.792
regnet_x_400mf                    72.834          90.950
regnet_x_800mf                    75.212          92.348
regnet_x_1_6gf                    77.040          93.440
regnet_x_3_2gf                    78.364          93.992
regnet_x_8gf                      79.344          94.686 
regnet_x_16gf                     80.058          94.944
regnet_x_32gf                     80.622          95.248
regnet_y_400mf                    74.046          91.716
regnet_y_800mf                    76.420          93.136
regnet_y_1_6gf                    77.950          93.966
regnet_y_3_2gf                    78.948          94.576
regnet_y_8gf                      80.032          95.048
regnet_y_16gf                     80.424          95.240
regnet_y_32gf                     80.878          95.340
vit_b_16                          81.072          95.318
vit_b_32                          75.912          92.466
vit_l_16                          79.662          94.638
vit_l_32                          76.972          93.070
convnext_tiny                     82.520          96.146
convnext_small                    83.616          96.650
convnext_base                     84.062          96.870
convnext_large                    84.414          96.976
================================  =============   =============


.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
.. _DenseNet: https://arxiv.org/abs/1608.06993
.. _Inception: https://arxiv.org/abs/1512.00567
.. _GoogLeNet: https://arxiv.org/abs/1409.4842
.. _ShuffleNet: https://arxiv.org/abs/1807.11164
.. _MobileNetV2: https://arxiv.org/abs/1801.04381
.. _MobileNetV3: https://arxiv.org/abs/1905.02244
.. _ResNeXt: https://arxiv.org/abs/1611.05431
.. _MNASNet: https://arxiv.org/abs/1807.11626
.. _EfficientNet: https://arxiv.org/abs/1905.11946
.. _RegNet: https://arxiv.org/abs/2003.13678
.. _VisionTransformer: https://arxiv.org/abs/2010.11929
.. _ConvNeXt: https://arxiv.org/abs/2201.03545

.. currentmodule:: torchvision.models

Alexnet
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    alexnet

VGG
---

.. autosummary::
    :toctree: generated/
    :template: function.rst

    vgg11
    vgg11_bn
    vgg13
    vgg13_bn
    vgg16
    vgg16_bn
    vgg19
    vgg19_bn


ResNet
------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152

SqueezeNet
----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    squeezenet1_0
    squeezenet1_1

DenseNet
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    densenet121
    densenet169
    densenet161
    densenet201

Inception v3
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    inception_v3

GoogLeNet
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    googlenet

ShuffleNet v2
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    shufflenet_v2_x1_5
    shufflenet_v2_x2_0

MobileNet v2
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mobilenet_v2

MobileNet v3
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mobilenet_v3_large
    mobilenet_v3_small

ResNext
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnext50_32x4d
    resnext101_32x8d

Wide ResNet
-----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    wide_resnet50_2
    wide_resnet101_2

MNASNet
--------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mnasnet0_5
    mnasnet0_75
    mnasnet1_0
    mnasnet1_3

EfficientNet
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    efficientnet_b0
    efficientnet_b1
    efficientnet_b2
    efficientnet_b3
    efficientnet_b4
    efficientnet_b5
    efficientnet_b6
    efficientnet_b7
    efficientnet_v2_s
    efficientnet_v2_m
    efficientnet_v2_l

RegNet
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    regnet_y_400mf
    regnet_y_800mf
    regnet_y_1_6gf
    regnet_y_3_2gf
    regnet_y_8gf
    regnet_y_16gf
    regnet_y_32gf
    regnet_y_128gf
    regnet_x_400mf
    regnet_x_800mf
    regnet_x_1_6gf
    regnet_x_3_2gf
    regnet_x_8gf
    regnet_x_16gf
    regnet_x_32gf

VisionTransformer
-----------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    vit_b_16
    vit_b_32
    vit_l_16
    vit_l_32

ConvNeXt
--------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    convnext_tiny
    convnext_small
    convnext_base
    convnext_large

Quantized Models
----------------

The following architectures provide support for INT8 quantized models. You can get
a model with random weights by calling its constructor:

.. code:: python

    import torchvision.models as models
    googlenet = models.quantization.googlenet()
    inception_v3 = models.quantization.inception_v3()
    mobilenet_v2 = models.quantization.mobilenet_v2()
    mobilenet_v3_large = models.quantization.mobilenet_v3_large()
    resnet18 = models.quantization.resnet18()
    resnet50 = models.quantization.resnet50()
    resnext101_32x8d = models.quantization.resnext101_32x8d()
    shufflenet_v2_x0_5 = models.quantization.shufflenet_v2_x0_5()
    shufflenet_v2_x1_0 = models.quantization.shufflenet_v2_x1_0()
    shufflenet_v2_x1_5 = models.quantization.shufflenet_v2_x1_5()
    shufflenet_v2_x2_0 = models.quantization.shufflenet_v2_x2_0()

Obtaining a pre-trained quantized model can be done with a few lines of code:

.. code:: python

    import torchvision.models as models
    model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    model.eval()
    # run the model with quantized inputs and weights
    out = model(torch.rand(1, 3, 224, 224))

We provide pre-trained quantized weights for the following models:

================================  =============  =============
Model                             Acc@1          Acc@5
================================  =============  =============
MobileNet V2                      71.658         90.150
MobileNet V3 Large                73.004         90.858
ShuffleNet V2 x1.0                68.360         87.582
ShuffleNet V2 x0.5                57.972         79.780
ResNet 18                         69.494         88.882
ResNet 50                         75.920         92.814
ResNext 101 32x8d                 78.986         94.480
Inception V3                      77.176         93.354
GoogleNet                         69.826         89.404
================================  =============  =============


Semantic Segmentation
=====================

The models subpackage contains definitions for the following model
architectures for semantic segmentation:

- `FCN ResNet50, ResNet101 <https://arxiv.org/abs/1411.4038>`_
- `DeepLabV3 ResNet50, ResNet101, MobileNetV3-Large <https://arxiv.org/abs/1706.05587>`_
- `LR-ASPP MobileNetV3-Large <https://arxiv.org/abs/1905.02244>`_

As with image classification models, all pre-trained models expect input images normalized in the same way.
The images have to be loaded in to a range of ``[0, 1]`` and then normalized using
``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
They have been trained on images resized such that their minimum size is 520.

For details on how to plot the masks of such models, you may refer to :ref:`semantic_seg_output`.

The pre-trained models have been trained on a subset of COCO train2017, on the 20 categories that are
present in the Pascal VOC dataset. You can see more information on how the subset has been selected in
``references/segmentation/coco_utils.py``. The classes that the pre-trained model outputs are the following,
in order:

  .. code-block:: python

      ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

The accuracies of the pre-trained models evaluated on COCO val2017 are as follows

================================  =============  ====================
Network                           mean IoU       global pixelwise acc
================================  =============  ====================
FCN ResNet50                      60.5           91.4
FCN ResNet101                     63.7           91.9
DeepLabV3 ResNet50                66.4           92.4
DeepLabV3 ResNet101               67.4           92.4
DeepLabV3 MobileNetV3-Large       60.3           91.2
LR-ASPP MobileNetV3-Large         57.9           91.2
================================  =============  ====================


Fully Convolutional Networks
----------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.segmentation.fcn_resnet50
    torchvision.models.segmentation.fcn_resnet101


DeepLabV3
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.segmentation.deeplabv3_resnet50
    torchvision.models.segmentation.deeplabv3_resnet101
    torchvision.models.segmentation.deeplabv3_mobilenet_v3_large


LR-ASPP
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.segmentation.lraspp_mobilenet_v3_large

.. _object_det_inst_seg_pers_keypoint_det:

Object Detection, Instance Segmentation and Person Keypoint Detection
=====================================================================

The models subpackage contains definitions for the following model
architectures for detection:

- `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_
- `FCOS <https://arxiv.org/abs/1904.01355>`_
- `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_
- `RetinaNet <https://arxiv.org/abs/1708.02002>`_
- `SSD <https://arxiv.org/abs/1512.02325>`_
- `SSDlite <https://arxiv.org/abs/1801.04381>`_

The pre-trained models for detection, instance segmentation and
keypoint detection are initialized with the classification models
in torchvision.

The models expect a list of ``Tensor[C, H, W]``, in the range ``0-1``.
The models internally resize the images but the behaviour varies depending
on the model. Check the constructor of the models for more information. The
output format of such models is illustrated in :ref:`instance_seg_output`.


For object detection and instance segmentation, the pre-trained
models return the predictions of the following classes:

  .. code-block:: python

      COCO_INSTANCE_CATEGORY_NAMES = [
          '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
          'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
          'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
          'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
      ]


Here are the summary of the accuracies for the models trained on
the instances set of COCO train2017 and evaluated on COCO val2017.

======================================  =======  ========  ===========
Network                                 box AP   mask AP   keypoint AP
======================================  =======  ========  ===========
Faster R-CNN ResNet-50 FPN              37.0     -         -
Faster R-CNN MobileNetV3-Large FPN      32.8     -         -
Faster R-CNN MobileNetV3-Large 320 FPN  22.8     -         -
FCOS ResNet-50 FPN                      39.2     -         -
RetinaNet ResNet-50 FPN                 36.4     -         -
SSD300 VGG16                            25.1     -         -
SSDlite320 MobileNetV3-Large            21.3     -         -
Mask R-CNN ResNet-50 FPN                37.9     34.6      -
======================================  =======  ========  ===========

For person keypoint detection, the accuracies for the pre-trained
models are as follows

================================  =======  ========  ===========
Network                           box AP   mask AP   keypoint AP
================================  =======  ========  ===========
Keypoint R-CNN ResNet-50 FPN      54.6     -         65.0
================================  =======  ========  ===========

For person keypoint detection, the pre-trained model return the
keypoints in the following order:

  .. code-block:: python

    COCO_PERSON_KEYPOINT_NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

Runtime characteristics
-----------------------

The implementations of the models for object detection, instance segmentation
and keypoint detection are efficient.

In the following table, we use 8 GPUs to report the results. During training,
we use a batch size of 2 per GPU for all models except SSD which uses 4
and SSDlite which uses 24. During testing a batch size  of 1 is used.

For test time, we report the time for the model evaluation and postprocessing
(including mask pasting in image), but not the time for computing the
precision-recall.

======================================  ===================  ==================  ===========
Network                                 train time (s / it)  test time (s / it)  memory (GB)
======================================  ===================  ==================  ===========
Faster R-CNN ResNet-50 FPN              0.2288               0.0590              5.2
Faster R-CNN MobileNetV3-Large FPN      0.1020               0.0415              1.0
Faster R-CNN MobileNetV3-Large 320 FPN  0.0978               0.0376              0.6
FCOS ResNet-50 FPN                      0.1450               0.0539              3.3
RetinaNet ResNet-50 FPN                 0.2514               0.0939              4.1
SSD300 VGG16                            0.2093               0.0744              1.5
SSDlite320 MobileNetV3-Large            0.1773               0.0906              1.5
Mask R-CNN ResNet-50 FPN                0.2728               0.0903              5.4
Keypoint R-CNN ResNet-50 FPN            0.3789               0.1242              6.8
======================================  ===================  ==================  ===========


Faster R-CNN
------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.fasterrcnn_resnet50_fpn
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn

FCOS
----

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.fcos_resnet50_fpn


RetinaNet
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.retinanet_resnet50_fpn


SSD
---

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.ssd300_vgg16


SSDlite
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.ssdlite320_mobilenet_v3_large


Mask R-CNN
----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.maskrcnn_resnet50_fpn


Keypoint R-CNN
--------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.detection.keypointrcnn_resnet50_fpn


Video classification
====================

We provide models for action recognition pre-trained on Kinetics-400.
They have all been trained with the scripts provided in ``references/video_classification``.

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB videos of shape (3 x T x H x W),
where H and W are expected to be 112, and T is a number of video frames in a clip.
The images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.43216, 0.394666, 0.37645]`` and ``std = [0.22803, 0.22145, 0.216989]``.


.. note::
  The normalization parameters are different from the image classification ones, and correspond
  to the mean and std from Kinetics-400.

.. note::
  For now, normalization code can be found in ``references/video_classification/transforms.py``,
  see the ``Normalize`` function there. Note that it differs from standard normalization for
  images because it assumes the video is 4d.

Kinetics 1-crop accuracies for clip length 16 (16x112x112)

================================  =============   =============
Network                           Clip acc@1      Clip acc@5
================================  =============   =============
ResNet 3D 18                      52.75           75.45
ResNet MC 18                      53.90           76.29
ResNet (2+1)D                     57.50           78.81
================================  =============   =============


ResNet 3D
----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.video.r3d_18

ResNet Mixed Convolution
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.video.mc3_18

ResNet (2+1)D
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.video.r2plus1d_18

Optical flow
============

Raft
----

.. autosummary::
    :toctree: generated/
    :template: function.rst

    torchvision.models.optical_flow.raft_large
    torchvision.models.optical_flow.raft_small
