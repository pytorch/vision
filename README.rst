torchvision
===========

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

TorchVision requires PyTorch 1.4 or newer.

Anaconda:

.. code:: bash

    conda install torchvision -c pytorch

pip:

.. code:: bash

    pip install torchvision

From source:

.. code:: bash

    python setup.py install
    # or, for OSX
    # MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

By default, GPU support is built if CUDA is found and ``torch.cuda.is_available()`` is true.
It's possible to force building GPU support by setting ``FORCE_CUDA=1`` environment variable,
which is useful when building a docker image.

Image Backend
=============
Torchvision currently supports the following image backends:

* `Pillow`_ (default)

* `Pillow-SIMD`_ - a **much faster** drop-in replacement for Pillow with SIMD. If installed will be used as the default.

* `accimage`_ - if installed can be activated by calling :code:`torchvision.set_image_backend('accimage')`

.. _Pillow : https://python-pillow.org/
.. _Pillow-SIMD : https://github.com/uploadcare/pillow-simd
.. _accimage: https://github.com/pytorch/accimage

C++ API
=======
TorchVision also offers a C++ API that contains C++ equivalent of python models. 

Installation From source:

.. code:: bash

    mkdir build
    cd build
    # Add -DWITH_CUDA=on support for the CUDA if needed
    cmake ..
    make 
    make install

Once installed, the library can be accessed in cmake (after properly configuring ``CMAKE_PREFIX_PATH``) via the :code:`TorchVision::TorchVision` target:

.. code:: rest

	find_package(TorchVision REQUIRED)
	target_link_libraries(my-target PUBLIC TorchVision::TorchVision)

The ``TorchVision`` package will also automatically look for the ``Torch`` and ``pybind11`` packages and add them as dependencies to ``my-target``,
so make sure that they are also available to cmake via the ``CMAKE_PREFIX_PATH``.

Documentation
=============
You can find the API documentation on the pytorch website: http://pytorch.org/docs/master/torchvision/

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Disclaimer on Datasets
======================

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
