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
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T

ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img)
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


####################################
# Reading Videos Using Torchvision
# --------------------------------
# We will first read a video using torchvision's read_video.
# Alternatively one can use the new VideoReader API (if torchvision is built from source).


from torchvision.io import read_video
# video_path = img_path = os.path.join(ASSETS_DIRECTORY, "./basketball.mp4")
video_path = img_path = os.path.join('/Users/NicolasHug/Downloads', "./basketball_hd.mp4")

#########################
# Read video returns the video_frames, audio_frames and the metadata
# We will focus only on the video frames.
# Note that v_frames tensor is of shape ``(num_frames, height, width, num_channels)``.
# We will visualize the optical flow between the first two frames of the video.


frames, _, _ = read_video(video_path)
frames = frames.permute(0, 3, 1, 2)

img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[101], frames[151]])

plot(img1_batch)


#########################
# We do some preprocessing to transform the
# image into batches and to normalize the image.
#
print("ol")


# def preprocess(batch):
#     transforms = T.Compose([
#             T.ConvertImageDtype(torch.float32),
#             T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
#         ])
#     batch = transforms(batch)
#     return batch


# img1_batch = preprocess(img1_batch)
# img2_batch = preprocess(img2_batch)


# ####################################
# # Estimating Optical flow using RAFT
# # ----------------------------------
# #
# #

# from torchvision.models.optical_flow import raft_large

# model = raft_large(pretrained=True, progress=False)
# model = model.eval()
# flow_preds = model(img1_batch, img2_batch)
# flows = flow_preds[-1]

# ####################################
# # Visualizing optical flow
# # -------------------------
# # Torchvision provides the :func:`~torchvision.utils.flow_to_image` utlity to
# # convert a flow into an RGB image. It also supports batches of flows.

# from torchvision.utils import flow_to_image

# flow_imgs = flow_to_image(flows)

# # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
# img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

# imgs = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
# for e in imgs:
#     print(e[0].shape, e[1].shape)
# plot(imgs)
