import os
import csv
import torchvision.transforms.functional as F
from .folder import ImageFolder


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
        bb_target_crop (boolean, optional): Crops bounding box from image as target.
        bb_landmarks_csv (string, optional): path to downloaded bb landmarks. Required
            if ``bbox`` is in target_type or bb_target_crop is True.

    '''

    def __init__(self, root, target_type='id', transform=None,
                 target_transform=None, bb_crop=False, bb_landmarks_csv=None):
        super(VGGFace2, self).__init__(root, transform=transform,
                                       target_transform=target_transform)

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.bb_crop = bb_crop
        self.get_bbox = self.bb_crop or 'bbox' in self.target_type

        if self.get_bbox:
            self.bb_data = {}
            with open(bb_landmarks_csv, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for path, x, y, w, h in reader:
                    self.bb_data[path] = (int(x), int(y), int(w), int(h))

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.get_bbox:
            bbox = self.bb_data[os.path.join(self.root, path) + '.jpg']

        if self.bb_crop:
            x, y, w, h = bbox
            sample = F.crop(sample, x, y, h, w)

        target = []
        for t in self.target_type:
            if t == 'id':
                target.append(label)
            elif t == 'bbox':
                target.extend(bbox)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
