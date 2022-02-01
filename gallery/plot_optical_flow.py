"""
==============================
Optical Flow using Torchvision
==============================

Optical flow is a task consisting of estimating per pixel motion between two consecutive frames of a video.
We aim to find the displacement of all image pixels and calculate their motion vectors.
The following example illustrates how torchvision can be used in predicting as well as
visualizing optical flow.

"""

# sphinx_gallery_thumbnail_path = "../../gallery/assets/optical_flow_thumbnail.png"

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


####################################
# Reading Videos Using Torchvision
# --------------------------------
# We will first read a video using torchvision's read_video.
# Alternatively once can use the new VideoReader API (if torchvision is built from source) or
# open-cv VideoCapture method.


from torchvision.io import read_video
video_path = img_path = os.path.join(ASSETS_DIRECTORY, "./basketball.mp4")

#########################
# Read video returns the video_frames, audio_frames and the metadata
# We will focus only on the video frames.
# Note that v_frames tensor is of shape ``(num_frames, height, width, num_channels)``.
# We will visualize the optical flow between the first two frames of the video.


v_frames, a_frames, meta_data = read_video(video_path)
print(v_frames.shape)

frame_1 = v_frames[1, :, :, :]
frame_2 = v_frames[2, :, :, :]

print(frame_1.shape)

#########################
# Little preprocessing to convert the frames to (1, C, H, W).
#


def frame_preprocess(frame):
    frame = frame.permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    return frame


frame_1 = frame_preprocess(frame_1)
frame_2 = frame_preprocess(frame_2)

# Now both the frames are of shape (1, C, H, W)
print(frame_1.shape)
print(frame_2.shape)


####################################
# Estimating Optical flow using RAFT
# ----------------------------------
# Some text
#

from torchvision.models.optical_flow import raft_large

model = raft_large(pretrained=True, progress=False)
model = model.eval()
flow_preds = model(2 * frame_1 / 255 - 1, 2 * frame_2 / 255 - 1)
flow = flow_preds[-1]

####################################
# Visualizing optical flow
# -------------------------
# Torchvision provides ``flow_to_image`` utlity to visualize optical flow.
# It can be used to convert single images as well as batches to flow.


from torchvision.utils import flow_to_image
img = flow_to_image(flow)

img = img.squeeze(0)
show(img)
