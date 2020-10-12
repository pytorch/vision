from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from typing import Dict
import sys
from torchvision.io.image import read_image
import torchvision.transforms as T
from typing import List
import numpy as np

img_path = "../test/assets/grace_hopper_517x606.jpg"


# def draw_bounding_boxes(
#     image: torch.Tensor,
#     boxes: torch.Tensor,
#     labels: torch.Tensor,
#     label_names: List[int] = None,
#     colors: Dict[int, str] = None,
#     draw_labels: bool = True,
#     width: int = 1
# ) -> torch.Tensor:

#     """
#     Draws bounding boxes on given image.

#     Args:
#         image (Tensor): Tensor of shape (C x H x W)
#         bboxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
#         labels (Tensor): Tensor of size (N) Labels for each bounding boxes.
#         label_names (List): List containing labels excluding background.
#         colors (dict): Dict with key as label id and value as color name.
#         draw_labels (bool): If True draws label names on bounding boxes.
#         width (int): Width of bounding box.
#     """

    # Code co-contributed by sumanthratna

    # Currently works for (C x H x W) images, but I think we should extend.
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer

#     if not (torch.is_tensor(image)):
#         raise TypeError('tensor expected, got {}'.format(type(image)))

#     if label_names is not None:
#         # Since for our detection models class 0 is background
#         label_names.insert(0, "__background__")

#     ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

#     # Neceassary check since FRCNN returns boxes which have grad enabled.
#     if(boxes.requires_grad):
#         boxes = boxes.detach()

#     boxes = boxes.to('cpu').numpy().astype('int').tolist()
#     labels = labels.to('cpu').numpy().astype('int').tolist()

#     img_to_draw = Image.fromarray(ndarr)
#     draw = ImageDraw.Draw(img_to_draw)

#     for bbox, label in zip(boxes, labels):
#         if colors is None:
#             draw.rectangle(bbox, width=width)
#         else:
#             draw.rectangle(bbox, width=width, outline=colors[label])

#         if label_names is None:
#             draw.text((bbox[0], bbox[1]), str(label))
#         else:
#             if draw_labels is True:
#                 draw.text((bbox[0], bbox[1]), label_names[int(label)])

#     img_to_draw.show()
#     return torch.from_numpy(np.array(img_to_draw))


if __name__ == "__main__":
#     # img = torch.rand(3, 3, 226, 226)
#     # img = read_image(img_path)

#     label_names = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#     colors = {1: "blue", 32: "aqua", 84: "yellow", 16: "black", 38: "brown"}

#     # label_names.insert(0, "__background__")

#     # print(label_names)

    img = Image.open(img_path)
    img = T.ToTensor()(img)
    print(img.shape)
    print(img.dim())
    img = torch.unsqueeze(img, 0)
    print(img.shape)
    print(img.dim())
    print(img.shape[0])

    if(img.dim() == 4):
        if(img.shape[0] == 1):
            img = img.squeeze(0)
    print(img.shape)
    print(img.dim())
    print(img.shape[0])


#     print(img.shape)

#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model = model.eval()
#     out = model(img)
#     # print(out)

#     boxes = out[0]["boxes"]
#     labels = out[0]["labels"]

#     print(out)

    # # print(boxes)
    # print(boxes.shape)

    # # print(labels)
    # print(labels.shape)

    # # draw_bounding_boxes(img, boxes, labels,)

    # # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # # print(img)
    # # print(img.shape)
    # img = img.squeeze(0)
    # # print(img.shape)

    
    # ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # # print(ndarr)

    # print(ndarr.shape)

    # # img = Image.fromarray(ndarr)
    # # draw = ImageDraw.Draw(img)

    # # print(boxes)
    # if(boxes.requires_grad):
    #     boxes = boxes.detach()

    # boxes = boxes.to('cpu').numpy().astype('int').tolist()
    # print(boxes)

    # labels = labels.to('cpu').numpy().astype('int').tolist()
    # print(labels)

    # img_to_draw = Image.fromarray(ndarr)
    # draw = ImageDraw.Draw(img_to_draw)

    # # print(img_to_draw.shape)

    # width = 1
    # for bbox, label in zip(boxes, labels):
    #     draw.rectangle(bbox, outline="red", width=5)
    #     # draw.text((bbox[0], bbox[1]), str(label))
    #     draw.text((bbox[0], bbox[1]), label_names[label])
    #     # draw.text
    #     # print("Draw Bbox")
    # img_to_draw.show()

    # # out = Image.open(img_path)
    # # draw = ImageDraw.Draw(out)
    # # box = [12, 23, 45, 42]
    # # draw.rectangle(box, outline="red", width=5)
    # # out.show()

    # img = img.squeeze(0)
    # img_drawn = draw_bounding_boxes(img, boxes, labels, label_names, colors=colors)

