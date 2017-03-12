from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class VOCSegmentation(data.Dataset):
    """VOC Segmentation Dataset Object
    input and target are both images

    NOTE: need to address https://github.com/pytorch/vision/issues/9

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg: 'train', 'val', 'test').
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target image
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(
            self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id).convert('RGB')
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        channels (int): number of channels
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, channels, height, width):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a Tensor containing [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # [xmin, ymin, xmax, ymax]
            bndbox = []
            for i, cur_bb in enumerate(bbox):
                bb_sz = int(cur_bb.text) - 1
                # scale height or width
                bb_sz = bb_sz / width if i % 2 == 0 else bb_sz / height
                bndbox.append(bb_sz)

            label_ind = self.class_to_ind[name]
            bndbox.append(label_ind)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, ind]

        return res  # [[xmin, ymin, xmax, ymax, ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(
            self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.squeeze(0), target

    def __len__(self):
        return len(self.ids)

    def show(self, index, subparts=False):
        '''Shows an image with its ground truth boxes overlaid optionally

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
            subparts (bool, optional): whether or not to display subpart
            bboxes of ground truths
                (default: False)
        '''
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id).convert('RGB')
        draw = ImageDraw.Draw(img)
        i = 0
        bndboxs = []
        classes = dict()  # maps class name to a class number
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            name = obj.find('name').text.lower().strip()
            if name not in classes:
                classes[name] = i
                i += 1
            bndboxs.append((name, [int(bb.text) - 1 for bb in bbox]))
            if subparts and not obj.find('part') is None:
                for part in obj.iter('part'):
                    name = part.find('name').text.lower().strip()
                    bbox = part.find('bndbox')
                    bndboxs.append((name, [int(bb.text) - 1 for bb in bbox]))
                    if name not in classes:
                        classes[name] = i
                        i += 1
        for name, bndbox in bndboxs:
            color = COLORS[classes[name] % len(COLORS)]
            draw.rectangle(bndbox, outline=color)
            draw.text(bndbox[:2], name, fill=color)
        img.show()
        return img


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type([])):
                annos = [torch.Tensor(a) for a in tup]
                targets.append(torch.stack(annos, 0))

    return (torch.stack(imgs, 0), targets)
