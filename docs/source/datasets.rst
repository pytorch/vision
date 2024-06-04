.. _datasets:

Datasets
========

Torchvision provides many built-in datasets in the ``torchvision.datasets``
module, as well as utility classes for building your own datasets.

Built-in datasets
-----------------

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples in parallel using ``torch.multiprocessing`` workers.
For example: ::

    imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

.. currentmodule:: torchvision.datasets

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.
You can also create your own datasets using the provided :ref:`base classes <base_classes_datasets>`.

Image classification
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    Caltech101
    Caltech256
    CelebA
    CIFAR10
    CIFAR100
    Country211
    DTD
    EMNIST
    EuroSAT
    FakeData
    FashionMNIST
    FER2013
    FGVCAircraft
    Flickr8k
    Flickr30k
    Flowers102
    Food101
    GTSRB
    INaturalist
    ImageNet
    Imagenette
    KMNIST
    LFWPeople
    LSUN
    MNIST
    Omniglot
    OxfordIIITPet
    Places365
    PCAM
    QMNIST
    RenderedSST2
    SEMEION
    SBU
    StanfordCars
    STL10
    SUN397
    SVHN
    USPS

Image detection or segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    CocoDetection
    CelebA
    Cityscapes
    Kitti
    OxfordIIITPet
    SBDataset
    VOCSegmentation
    VOCDetection
    WIDERFace

Optical Flow
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    FlyingChairs
    FlyingThings3D
    HD1K
    KittiFlow
    Sintel

Stereo Matching
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    CarlaStereo
    Kitti2012Stereo
    Kitti2015Stereo
    CREStereo
    FallingThingsStereo
    SceneFlowStereo
    SintelStereo
    InStereo2k
    ETH3DStereo
    Middlebury2014Stereo

Image pairs
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    LFWPairs
    PhotoTour

Image captioning
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    CocoCaptions

Video classification
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    HMDB51
    Kinetics
    UCF101

Video prediction
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    MovingMNIST

.. _base_classes_datasets:

Base classes for custom datasets
--------------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DatasetFolder
    ImageFolder
    VisionDataset

Transforms v2
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    wrap_dataset_for_transforms_v2
