GPU Decoder
===========

GPU decoder depends on ffmpeg for demuxing, uses NVDECODE APIs from the nvidia-video-codec sdk and uses cuda for processing on gpu. In order to use this, please follow the following steps:

* Download the latest `nvidia-video-codec-sdk <https://developer.nvidia.com/nvidia-video-codec-sdk/download>`_
* Extract the zipped file.
* Set TORCHVISION_INCLUDE environment variable to the location of the video codec headers(`nvcuvid.h` and `cuviddec.h`), which would be under `Interface` directory.
* Set TORCHVISION_LIBRARY environment variable to the location of the video codec library(`libnvcuvid.so`), which would be under `Lib/linux/stubs/x86_64` directory.
* Install the latest ffmpeg from `conda-forge` channel.

.. code:: bash

    conda install -c conda-forge ffmpeg

* Set CUDA_HOME environment variable to the cuda root directory.
* Build torchvision from source:

.. code:: bash

    python setup.py install
