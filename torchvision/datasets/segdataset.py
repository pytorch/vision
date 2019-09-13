from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from torchvision import utils, transforms
from PIL import Image
from torchvision.transforms import functional


class SegDataset(Dataset):
    """Segmentation Dataset: A dataset generator for segmentation tasks. """

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None, imagecolormode='rgb', maskcolormode='grayscale'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Masks
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform or list of transforms to be applied on a sample.
            seed (int): Specify a seed for the train and test split
            fraction (float): A float value from 0 to 1 which specifies the validation split fraction
            subset (string): 'Train' or 'Test' to select the appropriate set.
            imagecolormode (string): 'rgb' or 'grayscale' 
            maskcolormode (string): 'rgb' or 'grayscale' 
        """
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert(imagecolormode in ['rgb', 'grayscale'])
        assert(maskcolormode in ['rgb', 'grayscale'])

        self.imagecolorflag = self.color_dict[imagecolormode]
        self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        if not fraction:
            self.image_names = sorted(
                glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
            self.mask_names = sorted(
                glob.glob(os.path.join(self.root_dir, maskFolder, '*')))
        else:
            assert(subset in ['Train', 'Test'])
            self.fraction = fraction
            self.image_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, imageFolder, '*'))))
            self.mask_list = np.array(
                sorted(glob.glob(os.path.join(self.root_dir, maskFolder, '*'))))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == 'Train':
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list)*(1-self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list)*(1-self.fraction)))]
            else:
                self.image_names = self.image_list[int(
                    np.ceil(len(self.image_list)*(1-self.fraction))):]
                self.mask_names = self.mask_list[int(
                    np.ceil(len(self.mask_list)*(1-self.fraction))):]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if self.imagecolorflag:
            image = Image.open(img_name).convert('RGB')
        else:
            image = Image.open(img_name).convert('L')
        msk_name = self.mask_names[idx]
        if self.maskcolorflag:
            mask = Image.open(msk_name).convert('RGB')
        else:
            mask = Image.open(msk_name).convert('L')
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define few transformations for the Segmentation Dataloader


class Resize(object):
    """Resize image and masks. 
    size (sequence or int) – Desired output size. If size is a sequence like (h, w), output size will be matched to this. 
    If size is an int, smaller edge of the image will be matched to this number. 
    i.e, if height > width, then image will be rescaled to (size * height / width, size)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': functional.resize(image,self.size),
                'mask': functional.resize(mask,self.size)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': functional.to_tensor(image),
                'mask': functional.to_tensor(mask)}


class Normalize(object):
    '''Normalize image in the range 0 to 1 by dividing by 255.'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': self.lambd(image.type(torch.FloatTensor)),
                'mask': self.lambd(mask.type(torch.FloatTensor))}

    def __repr__(self):
        return self.__class__.__name__ + '()'


# Helper functions for creating dataloaders from either one dataset directory or separate 'Train' and 'Test' directories.


def get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4):
    """
        Create Train and Test dataloaders from two separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
        --Test
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
    """
    data_transforms = {
        'Train': transforms.Compose([ ToTensor(), Normalize()]),
        'Test': transforms.Compose([ ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(root_dir=os.path.join(data_dir, x),
                                    transform=data_transforms[x], maskFolder=maskFolder, imageFolder=imageFolder)
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=8)
                   for x in ['Train', 'Test']}
    return dataloaders


def get_dataloader_single_folder(data_dir, imageFolder='Images', maskFolder='Masks', fraction=0.2, batch_size=4):
    """
        Create training and testing dataloaders from a single folder.
        data_dir
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Masks1
        ---------MasksN

    """
    data_transforms = {
        'Train': transforms.Compose([ToTensor(), Normalize()]),
        'Test': transforms.Compose([ToTensor(), Normalize()]),
    }

    image_datasets = {x: SegDataset(data_dir, imageFolder=imageFolder, maskFolder=maskFolder, seed=100, fraction=fraction, subset=x, transform=data_transforms[x])
                      for x in ['Train', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=8)
                   for x in ['Train', 'Test']}
    return dataloaders
