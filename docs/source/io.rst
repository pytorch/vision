Decoding / Encoding images and videos
=====================================

.. currentmodule:: torchvision.io

The :mod:`torchvision.io` module provides utilities for decoding and encoding
images and videos.

Image Decoding
--------------

Torchvision currently supports decoding JPEG, PNG, WEBP, GIF, AVIF, and HEIC
images. JPEG decoding can also be done on CUDA GPUs.

The main entry point is the :func:`~torchvision.io.decode_image` function, which
you can use as an alternative to ``PIL.Image.open()``. It will decode images
straight into image Tensors, thus saving you the conversion and allowing you to
run transforms/preproc natively on tensors.

.. code::

    from torchvision.io import decode_image

    img = decode_image("path_to_image", mode="RGB")
    img.dtype  # torch.uint8

    # Or
    raw_encoded_bytes = ...  # read encoded bytes from your file system
    img = decode_image(raw_encoded_bytes, mode="RGB")


:func:`~torchvision.io.decode_image` will automatically detect the image format,
and call the corresponding decoder (except for HEIC and AVIF images, see details
in :func:`~torchvision.io.decode_avif` and :func:`~torchvision.io.decode_heic`).
You can also use the lower-level format-specific decoders which can be more
powerful, e.g. if you want to encode/decode JPEGs on CUDA.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    decode_image
    decode_jpeg
    decode_png
    decode_webp
    decode_avif
    decode_heic
    decode_gif

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

Video - DEPREACTED
------------------

.. warning::

    DEPRECATED: All the video decoding and encoding capabilities of torchvision
    are deprecated from version 0.22 and will be removed in version 0.24.  We
    recommend that you migrate to
    `TorchCodec <https://github.com/pytorch/torchcodec>`__, where we'll
    consolidate the future decoding/encoding capabilities of PyTorch

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
