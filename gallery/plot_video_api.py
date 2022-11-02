"""
=======================
Video API
=======================

This example illustrates some of the APIs that torchvision offers for
videos, together with the examples on how to build datasets and more.
"""

####################################
# 1. Introduction: building a new video object and examining the properties
# -------------------------------------------------------------------------
# First we select a video to test the object out. For the sake of argument
# we're using one from kinetics400 dataset.
# To create it, we need to define the path and the stream we want to use.

######################################
# Chosen video statistics:
#
# - WUzgd7C1pWA.mp4
#     - source:
#         - kinetics-400
#     - video:
#         - H-264
#         - MPEG-4 AVC (part 10) (avc1)
#         - fps: 29.97
#     - audio:
#         - MPEG AAC audio (mp4a)
#         - sample rate: 48K Hz
#

import torch
import torchvision
from torchvision.datasets.utils import download_url

# Download the sample video
download_url(
    "https://github.com/pytorch/vision/blob/main/test/assets/videos/WUzgd7C1pWA.mp4?raw=true",
    ".",
    "WUzgd7C1pWA.mp4"
)
video_path = "./WUzgd7C1pWA.mp4"

######################################
# Streams are defined in a similar fashion as torch devices. We encode them as strings in a form
# of ``stream_type:stream_id`` where ``stream_type`` is a string and ``stream_id`` a long int.
# The constructor accepts passing a ``stream_type`` only, in which case the stream is auto-discovered.
# Firstly, let's get the metadata for our particular video:

stream = "video"
video = torchvision.io.VideoReader(video_path, stream)
video.get_metadata()

######################################
# Here we can see that video has two streams - a video and an audio stream.
# Currently available stream types include ['video', 'audio'].
# Each descriptor consists of two parts: stream type (e.g. 'video') and a unique stream id
# (which are determined by video encoding).
# In this way, if the video container contains multiple streams of the same type,
# users can access the one they want.
# If only stream type is passed, the decoder auto-detects first stream of that type and returns it.

######################################
# Let's read all the frames from the video stream. By default, the return value of
# ``next(video_reader)`` is a dict containing the following fields.
#
# The return fields are:
#
# - ``data``: containing a torch.tensor
# - ``pts``: containing a float timestamp of this particular frame

metadata = video.get_metadata()
video.set_current_stream("audio")

frames = []  # we are going to save the frames here.
ptss = []  # pts is a presentation timestamp in seconds (float) of each frame
for frame in video:
    frames.append(frame['data'])
    ptss.append(frame['pts'])

print("PTS for first five frames ", ptss[:5])
print("Total number of frames: ", len(frames))
approx_nf = metadata['audio']['duration'][0] * metadata['audio']['framerate'][0]
print("Approx total number of datapoints we can expect: ", approx_nf)
print("Read data size: ", frames[0].size(0) * len(frames))

######################################
# But what if we only want to read certain time segment of the video?
# That can be done easily using the combination of our ``seek`` function, and the fact that each call
# to next returns the presentation timestamp of the returned frame in seconds.
#
# Given that our implementation relies on python iterators,
# we can leverage itertools to simplify the process and make it more pythonic.
#
# For example, if we wanted to read ten frames from second second:

# FIXME: With https://github.com/pytorch/vision/pull/6598 this blocks leads to sphinx build hanging when using
#  multiprocessing.
video.set_current_stream("video")

for _ in video.seek(2):
    pass
