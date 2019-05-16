torchvision
============

.. image:: https://travis-ci.org/pytorch/vision.svg?branch=master
    :target: https://travis-ci.org/pytorch/vision

.. image:: https://codecov.io/gh/pytorch/vision/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytorch/vision

.. image:: https://pepy.tech/badge/torchvision
    :target: https://pepy.tech/project/torchvision

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://pytorch.org/docs/stable/torchvision/index.html


The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.

Installation
============

Anaconda:

.. code:: bash

    conda install torchvision -c pytorch

pip:

.. code:: bash

    pip install torchvision

From source:

.. code:: bash

    python setup.py install


Image Backend
=============
Torchvision currently supports the following image backends:

* `Pillow`_ (default)

* `Pillow-SIMD`_ - a **much faster** drop-in replacement for Pillow with SIMD. If installed will be used as the default.

* `accimage`_ - if installed can be activated by calling :code:`torchvision.set_image_backend('accimage')`

.. _Pillow : https://python-pillow.org/
.. _Pillow-SIMD : https://github.com/uploadcare/pillow-simd
.. _accimage: https://github.com/pytorch/accimage

Documentation
=============
You can find the API documentation on the pytorch website: http://pytorch.org/docs/master/torchvision/

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.
