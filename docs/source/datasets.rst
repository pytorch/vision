Datasets
========

Torchvision provides many built-in datasets in the ``torchvision.datasets``
module, as well as utility classes for building your own datasets.

Built-in datasets
~~~~~~~~~~~~~~~~~

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


.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    Caltech101
    Caltech256
    CelebA
    CIFAR10
    CIFAR100
    Cityscapes
    CocoCaptions
    CocoDetection
    Country211
    DTD
    EMNIST
    FakeData
    FashionMNIST
    FER2013
    Flickr8k
    Flickr30k
    Flowers102
    FlyingChairs
    FlyingThings3D
    Food101
    FGVCAircraft
    GTSRB
    HD1K
    HMDB51
    ImageNet
    INaturalist
    Kinetics400
    Kitti
    KittiFlow
    KMNIST
    LFWPeople
    LFWPairs
    LSUN
    MNIST
    Omniglot
    OxfordIIITPet
    PCAM
    PhotoTour
    Places365
    QMNIST
    SBDataset
    SBU
    SEMEION
    Sintel
    STL10
    SUN397
    SVHN
    UCF101
    USPS
    VOCSegmentation
    VOCDetection
    WIDERFace

.. _base_classes_datasets:

Base classes for custom datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DatasetFolder
    ImageFolder
    VisionDataset
