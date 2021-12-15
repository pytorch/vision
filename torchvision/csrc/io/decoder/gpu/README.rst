GPU Decoder
===========

GPU decoder depends on ffmpeg for demuxing, uses NVDECODE APIs from the nvidia-video-codec sdk and uses cuda for processing on gpu. In order to use this, please follow the following steps:

* Download the latest `nvidia-video-codec-sdk <https://developer.nvidia.com/nvidia-video-codec-sdk/download>`_
* Extract the zipped file and copy the header files and libraries.

.. code:: bash

sudo cp Interface/* /usr/local/include/
sudo cp Lib/linux/stubs/x86_64/libnv* /usr/local/lib/

* Install ffmpeg and make sure ffmpeg headers and libraries are present under /usr/local/include and /usr/local/lib respectively.
* Set CUDA_HOME environment variable to the cuda root directory.
* Build torchvision from source:

.. code:: bash

    python setup.py install
