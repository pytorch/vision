Decoding / Encoding images and videos
=====================================

.. currentmodule:: torchvision.io

The :mod:`torchvision.io` module provides utilities for decoding and encoding
images and videos.

Image Decoding
--------------

Torchvision currently supports decoding JPEG, PNG, WEBP and GIF images. JPEG
decoding can also be done on CUDA GPUs.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decode_image
    decode_jpeg
    encode_png
    decode_gif
    decode_webp

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ImageReadMode

Obsolete decoding function:

.. autosummary::
    :toctree: generated/
    :template: function.rst

    read_image

Image Encoding
--------------

For encoding, JPEG (cpu and CUDA) and PNG are supported.


.. autosummary::
    :toctree: generated/
    :template: function.rst

    encode_jpeg
    write_jpeg
    encode_png
    write_png

IO operations
-------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    read_file
    write_file

Video
-----

.. warning::

    TODO recommend torchcodec

.. autosummary::
    :toctree: generated/
    :template: function.rst

    read_video
    read_video_timestamps
    write_video


**Fine-grained video API**

In addition to the :mod:`read_video` function, we provide a high-performance 
lower-level API for more fine-grained control compared to the :mod:`read_video` function.
It does all this whilst fully supporting torchscript.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    VideoReader
