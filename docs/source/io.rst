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


Fine-grained video API
-------------------

In addition to the :mod:`read_video` function, we provide a high-performance 
lower-level API for more fine-grained control compared to the :mod:`read_video` function.
It does all this whilst fully supporting torchscript.

.. autoclass:: VideoReader
    :members: __next__, get_metadata, set_current_stream, seek


Example of inspecting a video:

.. code:: python

    import torchvision
    video_path = "path to a test video"
    # Constructor allocates memory and a threaded decoder
    # instance per video. At the momet it takes two arguments:
    # path to the video file, and a wanted stream.
    reader = torchvision.io.VideoReader(video_path, "video")

    # The information about the video can be retrieved using the 
    # `get_metadata()` method. It returns a dictionary for every stream, with
    # duration and other relevant metadata (often frame rate)
    reader_md = reader.get_metadata()

    # metadata is structured as a dict of dicts with following structure
    # {"stream_type": {"attribute": [attribute per stream]}}
    #
    # following would print out the list of frame rates for every present video stream
    print(reader_md["video"]["fps"])

    # we explicitly select the stream we would like to operate on. In
    # the constructor we select a default video stream, but
    # in practice, we can set whichever stream we would like 
    video.set_current_stream("video:0")


Image
-----

.. autofunction:: read_image

.. autofunction:: decode_image

.. autofunction:: encode_jpeg

.. autofunction:: write_jpeg

.. autofunction:: encode_png

.. autofunction:: write_png
