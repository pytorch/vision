from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from .utils import download_url, check_integrity

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]
         }

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in [f for f in self.split_list] + ['train-extra','download-all']:
            raise ValueError('Wrong split entered! valid choices include "train"'
                              ', "train-extra", "extra", "download-all", and "test"')

        self.urls=[]
        self.filenames=[]
        self.file_md5s=[]  
        

        if split=='download-all':    
          for splt in self.split_list:
              self.urls.append(self.split_list[splt][0])
              self.filenames.append(self.split_list[splt][1])
              self.file_md5s.append(self.split_list[splt][2])  
          self.download()
          return  

        if split=='train-extra':
          for splt in self.split_list:
            if(splt != 'test'): 
              #print(splt)
              self.urls.append(self.split_list[splt][0])
              self.filenames.append(self.split_list[splt][1])
              self.file_md5s.append(self.split_list[splt][2])
            
        else:
          self.urls.append(self.split_list[split][0])
          self.filenames.append(self.split_list[split][1])
          self.file_md5s.append(self.split_list[split][2])
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio
        
        self.data=np.empty((32,32,3,0))
        self.labels=np.empty(0)
        for i in range(len(self.filenames)):
          # reading(loading) mat file as array
          loaded_mat = sio.loadmat(os.path.join(self.root, self.filenames[i]))
          self.data = np.concatenate((self.data, loaded_mat['X']), axis=3)
          
          # loading from the .mat file gives an np array of type np.uint8
          # converting to np.int64, so that we have a LongTensor after
          # the conversion from the numpy array
          # the squeeze is needed to obtain a 1D tensor
          y = loaded_mat['y'].astype(np.int64).squeeze()
          self.labels = np.concatenate((self.labels, y),axis=0)

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for i in range(len(self.filenames)):
          md5 = self.file_md5s[i]
          fpath = os.path.join(root, self.filenames[i])
        return check_integrity(fpath, md5)

    def download(self):
      for i in range(len(self.filenames)):
        md5 = self.file_md5s[i]
        download_url(self.urls[i], self.root, self.filenames[i], md5)
    
    def create_dataset(self, output_dir):
      """
      Creates and configures a dataset in the given directory for full training and test sets. 
      The training folder will contain both 'train' and 'extra' splits
      
      Args: 
          output_dir (string) : The directory in which the dataset will be created

      """
      import torch.utils.data as data
      from PIL import Image
      import scipy.io as sio
      from pathlib import Path
      import numpy as np 

      source_dir = self.root
      save_directory = output_dir
      training_dir = os.path.join(output_dir,'train-extra')
      test_dir = os.path.join(output_dir,'test')
     
      if os.path.isdir(training_dir):
        return training_dir, test_dir
      else:  
        for matfile in [f for f in os.listdir(source_dir) if f.endswith('.mat')]: 
          # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(source_dir, matfile))
            # The matrix shape is (32,32,3,# of samples)
            for idx in range(loaded_mat['X'].shape[3]):   
              img = Image.fromarray(loaded_mat['X'][:,:,:,idx]).convert('RGB')

              # loading from the .mat file gives an np array of type np.uint8
              # converting to np.int64, so that we have a LongTensor after
              # the conversion from the numpy array
              # the squeeze is needed to obtain a 1D tensor
              labels = loaded_mat['y'].astype(np.int64).squeeze()
              # the svhn dataset assigns the class label "10" to the digit 0
              # this makes it inconsistent with several loss functions
              # which expect the class labels to be in the range [0, C-1]
              np.place(labels, labels == 10, 0)
              
              prefix = 'test' if 'test' in Path(matfile).stem else 'train-extra'
              image_name = '{0}/{1}/{2}_{3}.jpg'.format(prefix, labels[idx], Path(matfile).stem, idx)
              
              full_path = os.path.join(save_directory, image_name)
              print('{0} '.format(image_name))

              Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True) 

              img.save(full_path)
                    
      return training_dir, test_dir


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

