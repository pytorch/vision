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
    :members: next, get_metadata, set_current_stream, seek


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

A note on streams:
Each stream descriptor consists of two parts: stream type (e.g. 'video') and
a unique stream id (which are determined by the video encoding).
In this way, if the video contaner contains multiple
streams of the same type, users can acces the one they want.
If only stream type is passed, the decoder auto-detects first stream of that type.



More examples - video reading:

.. code:: python
    # if we want to simply read all frames, the reader
    # object behaves like an iterator, returning a dictionary
    # at each iteration. For example if we wanted to read all the frames:
    frames = []
    for frame in reader:
        frames.append(frame['data'])

    # this also means we can utilize :mod:`itertools` for 
    # more specific operations. Let's say we wanted to read
    # 10 frames after second second. This could easily be done by
    import itertools
    frames = []
    for frame in itertools.islice(reader.seek(2), 10):
        frames.append(frame['data'])

    # or if we wanted to read video from 2nd to 5th second
    frames = []
    for frame in itertools.takewhile(lambda x: x['pts'] <= 5, reader.seek(2)):
        frames.append(frame['data'])



Image
-----

.. autofunction:: read_image

.. autofunction:: decode_image

.. autofunction:: encode_jpeg

.. autofunction:: write_jpeg

.. autofunction:: encode_png

.. autofunction:: write_png
