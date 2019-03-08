torchvision.datasets
====================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
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


MNIST
~~~~~

.. autoclass:: MNIST

Fashion-MNIST
~~~~~~~~~~~~~

.. autoclass:: FashionMNIST

KMNIST
~~~~~~~~~~~~~

.. autoclass:: KMNIST

EMNIST
~~~~~~

.. autoclass:: EMNIST

FakeData
~~~~~~~~

.. autoclass:: FakeData

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

LSUN
~~~~

.. autoclass:: LSUN
  :members: __getitem__
  :special-members:

ImageFolder
~~~~~~~~~~~

.. autoclass:: ImageFolder
  :members: __getitem__
  :special-members:

DatasetFolder
~~~~~~~~~~~~~

.. autoclass:: DatasetFolder
  :members: __getitem__
  :special-members:



Imagenet-12
~~~~~~~~~~~

This should simply be implemented with an ``ImageFolder`` dataset.
The data is preprocessed `as described
here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__

`Here is an
example <https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126>`__.

CIFAR
~~~~~

.. autoclass:: CIFAR10
  :members: __getitem__
  :special-members:

.. autoclass:: CIFAR100

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

PhotoTour
~~~~~~~~~


.. autoclass:: PhotoTour
  :members: __getitem__
  :special-members:

SBU
~~~


.. autoclass:: SBU
  :members: __getitem__
  :special-members:

Flickr
~~~~~~


.. autoclass:: Flickr8k
  :members: __getitem__
  :special-members:

.. autoclass:: Flickr30k
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

Cityscapes
~~~~~~~~~~

.. note ::
    Requires Cityscape to be downloaded.

.. autoclass:: Cityscapes
  :members: __getitem__
  :special-members:
