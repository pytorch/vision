"""
==============================
Optical Flow using Torchvision
==============================

The following example illustrates ......


"""

# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail2.png"

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


def frame_preprocess(frame, device):
    frame = frame.permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


from torchvision.models.optical_flow import raft_large
model = raft_large(pretrained=True)

device = "cpu"
model = model.eval()

video_path = "./crowd.mp4"

from torchvision.io import read_video

v_frames, a_frames, meta_data = read_video(video_path)

print(v_frames.shape)

frame_1 = v_frames[1, :, :, :]
frame_2 = v_frames[2, :, :, :]

# print(frame_1.shape)

# show(frame_1)

frame_1 = frame_preprocess(frame_1, device)
frame_2 = frame_preprocess(frame_2, device)

print(frame_1.shape)
print(frame_2.shape)

flow_preds = model(2 * frame_1 / 255 - 1, 2 * frame_2 / 255 - 1)
flow_up = flow_preds[-1]

from torchvision.utils import flow_to_image
flo = flow_to_image(flow_up)
flo = flo.squeeze(0)

print(flo.shape)
print(frame_1.shape)

show([flo, frame_1])
plt.show()
