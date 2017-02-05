import torch.utils.data as data
from PIL import Image
import os.path

def make_dataset(file_name):
    images = []
    with open(file_name) as f:
        for line in f.read().splitlines():
            img, c = line.split()
            images.append((img, int(c)))
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageData(data.Dataset):
    """
    Caffe style ImageData

    Arguments:
        root: data directory
        list_file: a file name. Each row of the file: image_path class
    """
    def __init__(self, root, list_file, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(list_file)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
