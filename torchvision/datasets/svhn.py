from __future__ import print_function
import scipy.io as sio
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np

class SVHN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""
    
    def __init__(self, root, dataset='train', transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset  # training set or test set or extra set
        
        # download and load the data 
        if self.dataset=='train':
            self.url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
            self.filename = "train_32x32.mat"
            self.file_md5 = "e26dedcc434d2e4c54c9b2d4a06d8373"

            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
            self.train_data = []
            self.train_label = []
            
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(root, self.filename))
            
            self.train_data = loaded_mat['X']
            self.train_labels = loaded_mat['y']
            
            self.train_data = np.transpose(self.train_data,(3,2,1,0))
            
        elif self.dataset == 'extra':
            self.url = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
            self.filename = "extra_32x32.mat"
            self.file_md5 = "a93ce644f1a588dc4d68dda5feec44a7"

            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
            self.train_data = []
            self.train_label = []
            
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(root, self.filename))
            
            self.train_data = loaded_mat['X']
            self.train_labels = loaded_mat['y']
            
            self.train_data = np.transpose(self.train_data,(3,2,1,0))
            
        elif self.dataset == 'test':
            self.url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
            self.filename = "test_32x32.mat"
            self.file_md5 = "eb5a983be6a315427106f1b164d9cef3"
            
            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
            self.test_data = []
            self.test_label = []
            
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(root, self.filename))
            
            self.test_data = loaded_mat['X']
            self.test_labels = loaded_mat['y']
           
            self.test_data = np.transpose(self.test_data,(3,2,1,0))
            
        else:
            print ("Wrong dataset entered! Please use dataset=train or dataset=extra or dataset=test")
        

    def __getitem__(self, index):
        if self.dataset == 'train' or self.dataset == 'extra':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.dataset == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1,2,0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
     

    def __len__(self):
        if self.dataset == 'train':
            return 73257
        elif self.dataset == 'extra':
            return 531131
        elif self.dataset == 'test':
            return 26032


    def download(self):
        from six.moves import urllib
        import tarfile
        import hashlib

        root = self.root
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        print ("about to download")
        # downloads file
        if os.path.isfile(fpath):
            print('Using downloaded file: ' + fpath)
        else:
            print('Downloading ' + self.url + ' to ' + fpath)
            urllib.request.urlretrieve(self.url, fpath)
            print ('Downloaded!')

    def _check_integrity(self):
        import hashlib
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        md5c = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
        if md5c != md5:
                return False
        return True

