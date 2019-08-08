import torch
import torchvision

import transforms as T

class ConvertVOCtoCOCO(object):
    CLASSES = (
        "__background__", "aeroplane", "bicycle",
        "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )
    def __call__(self, image, target):
        # return image, target
        anno = target['annotations']
        filename = anno["filename"].split('.')[0]
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            ishard.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        ishard = torch.as_tensor(ishard)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8

        return image, target

class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super(VOCDetection, self).__init__(img_folder,  year, image_set)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        target = dict(image_id=idx, annotations=target['annotation'])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # img = img[[2, 1, 0],:]
        return img, target

def get_voc(root, image_set, transforms):
    t = [ConvertVOCtoCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = VOCDetection(img_folder=root, year='2007', image_set=image_set, transforms=transforms)

    return dataset
