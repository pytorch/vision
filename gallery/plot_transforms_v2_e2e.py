"""
==================================================
transforms v2: End-to-end object detection example
==================================================

"""

import pathlib
from collections import defaultdict

import PIL.Image

import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()


def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"])

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()


########################################################################################################################
# To use transforms v2, you can use the regular datasets and models and just have import transforms differently

from torchvision import models, datasets
import torchvision.transforms.v2 as transforms


########################################################################################################################
# Looks easy doesn't it? That's all there is to it. We can already start to load some data.


def load_example_coco_detection_dataset(**kwargs):
    root = pathlib.Path("assets") / "coco"
    return datasets.CocoDetection(str(root / "images"), str(root / "instances.json"), **kwargs)


dataset = load_example_coco_detection_dataset()

sample = dataset[0]
image, target = sample
print(type(image))
print(type(target), type(target[0]), list(target[0].keys()))


########################################################################################################################
# Transforms v2 dispatch works with datapoints. Here is how to get them

dataset = datasets.wrap_dataset_for_transforms_v2(dataset)

sample = dataset[0]
image, target = sample
print(type(image))
print(type(target), list(target.keys()))

show(sample)


########################################################################################################################
# Let's define a object detection pipeline

transform = transforms.Compose(
    [
        transforms.RandomPhotometricDistort(),
        transforms.RandomZoomOut(
            fill=defaultdict(
                lambda: 0,
                {
                    PIL.Image.Image: (123, 117, 104),
                },
            )
        ),
        transforms.RandomIoUCrop(),
        transforms.RandomHorizontalFlip(),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
        # FIXME: this is not good UX
        transforms.SanitizeBoundingBoxes(labels_getter=lambda sample: sample[1]["labels"]),
    ]
)

dataset = load_example_coco_detection_dataset(transforms=transform)
dataset = datasets.wrap_dataset_for_transforms_v2(dataset)

torch.manual_seed(0)
sample = dataset[0]
image, target = sample

show(sample)


########################################################################################################################
# Still works with the dataloader and can be used directly by our detection models

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

model = models.get_model("ssd300_vgg16", weights=None, weights_backbone=None).train()

for images, targets in data_loader:
    prediction = model(images, targets)
    print(prediction)
    # put your training logic here
    break
