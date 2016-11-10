# torch-vision

This repository consists of:

- `vision.datasets` : Data loaders for popular vision datasets
- `vision.transforms` : Common image transformations such as random crop, rotations etc.
- `[WIP] vision.models` : Model definitions and Pre-trained models for popular models such as AlexNet, VGG, ResNet etc.

# Installation

Binaries:

```bash
conda install pytorch-vision -c https://conda.anaconda.org/t/6N-MsQ4WZ7jo/soumith
```

From Source:

```bash
pip install -r requirements.txt
pip install .
```

# Datasets

Datasets have the API:
- `__getitem__`
- `__len__`
They all subclass from `torch.utils.data.Dataset`
Hence, they can all be multi-threaded (python multiprocessing) using standard torch.utils.data.DataLoader.

For example:

`torch.utils.data.DataLoader(coco_cap, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)`

In the constructor, each dataset has a slightly different API as needed, but they all take the keyword args:

- `transform` - a function that takes in an image and returns a transformed version
  - common stuff like `ToTensor`, `RandomCrop`, etc. These can be composed together with `transforms.Compose` (see transforms section below)
- `target_transform` - a function that takes in the target and transforms it. For example, take in the caption string and return a tensor of word indices.

The following datasets are available:

- COCO (Captioning and Detection)
- LSUN Classification
- Imagenet-12
- ImageFolder

### COCO

This requires the [COCO API to be installed](https://github.com/pdollar/coco/tree/master/PythonAPI)

#### Captions:

`dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])`

Example:

```python
import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'dir where images are', annFile = 'json annotation file', transform=transforms.toTensor)

print('Number of samples:', len(cap))
img, target = cap[3] # load 4th sample

print(img.size())
print(target)
```

Output:

```
```

#### Detection:
`dset.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])`

### LSUN

`dset.LSUN(db_path, classes='train', [transform, target_transform])`

- db_path = root directory for the database files
- classes =
  - 'train' - all categories, training set
  - 'val' - all categories, validation set
  - 'test' - all categories, test set
  - ['bedroom_train', 'church_train', ...] : a list of categories to load


### ImageFolder

A generic data loader where the images are arranged in this way:

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

`dset.ImageFolder(root="root folder path", [transform, target_transform])`

It has the members:

- `self.classes` - The class names as a list
- `self.class_to_idx` - Corresponding class indices
- `self.imgs` - The list of (image path, class-index) tuples


### Imagenet-12

This is simply implemented with an ImageFolder dataset, after the data is preprocessed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)
