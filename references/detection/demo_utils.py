#personal utils for pytorch demo
import os
import numpy as np
from PIL import ImageDraw, Image
import torch
import torch.utils.data
import torchvision

def get_boxes(mask_array, use_height_width_format = False):
    boxes = []
    # instances are encoded as different colors
    obj_ids = np.unique(mask_array)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)
    # split the color-encoded mask into a set
    # of binary masks
    masks = mask_array == obj_ids[:, None, None]
    for i in range(num_objs):
        pos = np.where(masks[i])
        if use_height_width_format:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            width = xmax - xmin
            height = ymax - ymin
            boxes.append([xmin, ymin, width, height])
        else:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def draw_prediction(img, prediction):
    img_final = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(img_final)
    boxes = prediction[0]['boxes'].cpu().numpy()
    for i in range(boxes.shape[0]):
      (xmin, ymin, xmax, ymax) = boxes[i]
      draw.rectangle([(xmin, ymin), (xmax, ymax)], outline ="red")

    return img_final

def draw_bounding_boxes(mask_path, img_path):
    colors = ['red', 'blue', 'green', 'purple', 'pink']
    #Open Image
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    boxes = get_boxes(mask)   # instances are encoded as different colors
    for i in range(len(boxes)):
        (xmin, ymin, xmax, ymax) = boxes[i]
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline = colors[i])
    return img

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "PNGImages", img_name)
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_name

    def __len__(self):
        return len(self.imgs)
