from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
from PIL import Image

class OMNIGLOT(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    Args:

    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    - input_is_filename: if True, the returned data is (filename,target), it is a pair (PIL.Image,target) elsewhere
    '''
    def __init__(self, root, transform=None, target_transform=None, download=False,input_is_filename=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.input_is_filename=input_is_filename
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        self.all_items=find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes=index_classes(self.all_items)

    def __getitem__(self, index):
        filename=self.all_items[index][0]
        path=str.join('/',[self.all_items[index][2],filename])

        if (not self.input_is_filename):
            img=Image.open(path).convert('RGB')
        else:
            img=path

        target=self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return  img,target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from "+file_path+" to "+file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")

def find_classes(root_dir):
    retour=[]
    for (root,dirs,files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r=root.split('/')
                lr=len(r)
                retour.append((f,r[lr-2]+"/"+r[lr-1],root))
    print("Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]]=len(idx)
    print("Found %d classes"% len(idx))
    return idx