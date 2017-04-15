import os
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10

class Cub2011(ImageFolder, CIFAR10):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    
    train_list = [
        ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', '4c84da568f89519f84640c54b7fba7c2'],
        ['002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg', 'e7db63424d0e384dba02aacaf298cdc0'],
    ]
    test_list = [
        ['198.Rock_Wren/Rock_Wren_0001_189289.jpg', '487d082f1fbd58faa7b08aa5ede3cc00'],
        ['200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg', '96fd60ce4b4805e64368efc32bf5c6fe']
    ]
    
    def __init__(self, root, transform=None, target_transform=None, download=False, **kwargs):
        self.root = root
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        ImageFolder.__init__(self, os.path.join(root, self.base_folder), transform = transform, target_transform = target_transform, **kwargs)
