torchvision.datasets
====================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: torchvision.datasets

CelebA
~~~~~~

.. autoclass:: CelebA
  :members: __getitem__
  :special-members:

CIFAR
~~~~~

.. autoclass:: CIFAR10
  :members: __getitem__
  :special-members:

.. autoclass:: CIFAR100

Cityscapes
~~~~~~~~~~

.. note ::
    Requires Cityscape to be downloaded.

.. autoclass:: Cityscapes
  :members: __getitem__
  :special-members:

COCO
~~~~

.. note ::
    These require the `COCO API to be installed`_

.. _COCO API to be installed: https://github.com/pdollar/coco/tree/master/PythonAPI


Captions
^^^^^^^^

.. autoclass:: CocoCaptions
  :members: __getitem__
  :special-members:


Detection
^^^^^^^^^

.. autoclass:: CocoDetection
  :members: __getitem__
  :special-members:

DatasetFolder
~~~~~~~~~~~~~

.. autoclass:: DatasetFolder
  :members: __getitem__
  :special-members:


EMNIST
~~~~~~

.. autoclass:: EMNIST

FakeData
~~~~~~~~

.. autoclass:: FakeData

Fashion-MNIST
~~~~~~~~~~~~~

.. autoclass:: FashionMNIST

Flickr
~~~~~~

.. autoclass:: Flickr8k
  :members: __getitem__
  :special-members:

.. autoclass:: Flickr30k
  :members: __getitem__
  :special-members:

HMDB51
~~~~~~~

.. autoclass:: HMDB51
  :members: __getitem__
  :special-members:

ImageFolder
~~~~~~~~~~~

.. autoclass:: ImageFolder
  :members: __getitem__
  :special-members:

ImageNet
~~~~~~~~~~~

.. autoclass:: ImageNet

.. note ::
    This requires `scipy` to be installed

Kinetics-400
~~~~~~~~~~~~

.. autoclass:: Kinetics400
  :members: __getitem__
  :special-members:

KMNIST
~~~~~~~~~~~~~

.. autoclass:: KMNIST

LSUN
~~~~

.. autoclass:: LSUN
  :members: __getitem__
  :special-members:

MNIST
~~~~~

.. autoclass:: MNIST

Omniglot
~~~~~~

.. autoclass:: Omniglot

PhotoTour
~~~~~~~~~

.. autoclass:: PhotoTour
  :members: __getitem__
  :special-members:

Places365
~~~~~~~~~

.. autoclass:: Places365
  :members: __getitem__
  :special-members:

QMNIST
~~~~~~

.. autoclass:: QMNIST

SBD
~~~~~~

.. autoclass:: SBDataset
  :members: __getitem__
  :special-members:

SBU
~~~

.. autoclass:: SBU
  :members: __getitem__
  :special-members:

STL10
~~~~~

.. autoclass:: STL10
  :members: __getitem__
  :special-members:

SVHN
~~~~~

.. autoclass:: SVHN
  :members: __getitem__
  :special-members:

UCF101
~~~~~~~

.. autoclass:: UCF101
  :members: __getitem__
  :special-members:

USPS
~~~~~

.. autoclass:: USPS
  :members: __getitem__
  :special-members:

VOC
~~~~~~

.. autoclass:: VOCSegmentation
  :members: __getitem__
  :special-members:

.. autoclass:: VOCDetection
  :members: __getitem__
  :special-members:

