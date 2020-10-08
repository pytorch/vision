torchvision.io
==============

.. currentmodule:: torchvision.io

The :mod:`torchvision.io` package provides functions for performing IO
operations. They are currently specific to reading and writing video and
images.

Video
-----

.. autofunction:: read_video

.. autofunction:: read_video_timestamps

.. autofunction:: write_video


Low-level video API
-------------------

In addition to the :mod:`read_video` function, we provide a high-performance 
low-level API for more fine-grained control compared to the old API.
We expose it to users using TORCHBIND C++ registration, and allow users
to access it via :mod:`torch.classes` import or via :mod:`torchvision.io`. For example

.. autoclass:: Video
    :members: next, get_metadata, set_current_stream, seek


Example of usage:

.. code:: python

    import torch
    import torchvision
    video_path = "path to a test video"
    # Constructor allocates memory and a threaded decoder
    # instance per video. At the momet it takes two arguments:
    # path to the video file, and a wanted stream.
    
    reader = torch.classes.torchvision.Video(video_path, "video")
    # equivalently, on could call
    reader = torchvision.io.Video(video_path, "video")

    # The information about the video can be retrieved using the 
    # `get_metadata()` method. It returns a dictionary for every stream, with
    # duration and other relevant metadata (often frame rate)
    reader_md = reader.get_metadata()

    # metadata is structured as a dict of dicts with following structure
    # {"stream_type": {"attribute": [attribute per stream]}}
    #
    # following would print out the list of frame rates for every present video stream
    print(reader_md["video"]["fps"])


Image
-----

.. autofunction:: read_image

.. autofunction:: decode_image

.. autofunction:: encode_jpeg

.. autofunction:: write_jpeg

.. autofunction:: encode_png

.. autofunction:: write_png
