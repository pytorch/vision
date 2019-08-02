import os
import csv
import torchvision.transforms.functional as F
from .folder import ImageFolder


def read_bbox_csv(root, csv_path):
    bb_data = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for path, x, y, w, h in reader:
            path = os.path.join(root, path) + '.jpg'
            bb_data[path] = (int(x), int(y), int(w), int(h))

    return bb_data


def read_landmark_csv(root, csv_path):
    landmark_data = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            path = os.path.join(root, row[0]) + '.jpg'
            landmarks = tuple(float(x) for x in row[1:])
            landmark_data[path] = landmarks

    return landmark_data


class VGGFace2(ImageFolder):
    '''`VGGFace2: A large scale image dataset for face recognition
        <http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/index.html>`_ Dataset.

    Args:
        root (string): Path to downloaded dataset.
        target_type (string or list, optional): Target type for each sample, ``id``
            or ``bbox``. Can also be a list to output a tuple with all specified
            target types.
            The targets represent:
                ``id`` (int): label/id for each person.
                ``bbox`` (tuple[int]) bounding box encoded as x, y, width, height
            Defaults to ``id``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        bbox_crop (boolean, optional): Crops bounding box from image as target.
        bbox_csv (string, optional): path to downloaded bounding box csv. Required
            if ``bbox`` is in target_type or bb_target_crop is True.
        landmark_csv (string, optional): path to downloaded landmarks csv. Required
            if ``landmark`` is in target_type.
    '''

    def __init__(self, root, target_type='id', transform=None,
                 target_transform=None, bbox_crop=False, bbox_csv=None,
                 landmark_csv=None):
        super(VGGFace2, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.bbox_crop = bbox_crop

        if self.bbox_crop or 'bbox' in self.target_type:
            if bbox_csv is None:
                raise ValueError("bbox_csv cannot be None if 'bbox' "
                                 "in target_type or bbox_crop=True")
            self.bb_data = read_bbox_csv(self.root, bbox_csv)

        if 'landmark' in target_type:
            if landmark_csv is None:
                raise ValueError("bbox_csv cannot be None if 'landmark' "
                                 "in target_type")
            self.landmark_data = read_landmark_csv(self.root, landmark_csv)

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.bbox_crop:
            x, y, w, h = self.bb_data[path]
            sample = F.crop(sample, x, y, h, w)

        target = []
        for t in self.target_type:
            if t == 'id':
                target.append(label)
            elif t == 'bbox':
                target.append(self.bb_data[path])
            elif t == 'landmark':
                target.append(self.landmark_data[path])

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, *target)

    def __len__(self):
        return len(self.samples)

    def extra_repr(self):
        return 'Target type: {}'.format(self.target_type)
