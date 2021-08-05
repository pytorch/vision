import torch
import numpy as np
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont, ImageColor
import torchvision.transforms.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    radius: Optional[int] = 4,
    connect: Optional[bool] = False,
    font: Optional[str] = None,
    font_size: int = 10
) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    out_dtype = torch.uint8

    img_kpts = keypoints.to(torch.int64).tolist()

    for i, kpt_inst in enumerate(img_kpts):
        # Iterate over every keypointpt in each keypoint instance.
        for kpt in kpt_inst:
            # print(kpt, kpt[0], kpt[1])
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            # print(x1, y1, x2, y2)
            draw.ellipse([x1, y1, x2, y2], fill="red", outline=None, width=0)

    im = np.array(img_to_draw)
    img = Image.fromarray(im)
    img.save("example3.jpg")

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=out_dtype)


if __name__ == '__main__':
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model = model.eval()
    IMAGE_PATH = 'demo_im3.jpg'
    image_tensor2 = read_image(IMAGE_PATH)
    image = Image.open(IMAGE_PATH)
    image_tensor = to_tensor(image)
    print(image_tensor.size())
    print(image_tensor.dtype)
    output = model([image_tensor])[0]

    kpts = output['keypoints']

    print(kpts)
    print(kpts.size())
    res = draw_keypoints(image_tensor2, kpts)
    # show(res)

    # print(output)
    # print(len(output))
