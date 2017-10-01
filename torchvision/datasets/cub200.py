from __future__ import print_function
import os
import errno
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

class CUB2002010(data.Dataset):
	urls	= [
			'http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz',
			'http://www.vision.caltech.edu/visipedia-data/CUB-200/lists.tgz'
		  ]
	raw_folder		= 'raw'
	processed_folder 	= 'processed'
	training_file		= 'training.pt'
	test_file		= 'test.pt'
	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		self.root		= os.path.expanduser(root)
		self.transform		= transform
		self.target_transform	= target_transform
		self.train		= train
		
		if download == True:
			self.download()
			
		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if self.train == True:
			self.train_data, self.train_labels	= torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
		else:
			self.test_data, self.test_labels	= torch.load(os.path.join(self.root, self.processed_folder, self.test_file))
		
	def __getitem__(self, index):
		if self.train == True:
			img, target	= self.train_data[index], self.train_labels[index]
		else:
			img, target	= self.test_data[index], self.test_labels[index]
			
		img	= Image.fromarray(img.numpy(), mode='L')
		
		if self.transform is not None:
			img	= self.transform(img)
		
		if self.target_transform is not None:
			img	= self.target_transform(img)
			
		return img, target
		
	def __len__(self):
		if self.train == True:
			return len(self.train_data)
		else:
			return len(self.test_data)
			
	def _check_exists(self):
		return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file))
		
	def download(self):
		from six.moves import urllib
		import tarfile
		
		if self._check_exists():
			return
			
		try:
			os.makedirs(os.path.join(self.root, self.raw_folder))
			os.makedirs(os.path.join(self.root, self.processed_folder))
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise
			
		for url in self.urls:
			print('Downloading ' + url)
			data		= urllib.request.urlopen(url)
			filename	= url.rpartition('/')[2]
			file_path	= os.path.join(self.root, self.raw_folder, filename)
			with open(file_path, 'wb') as f:
				f.write(data.read())
			tar	= tarfile.open(file_path, 'r')
			for item in tar:
				tar.extract(item, file_path.replace(filename, ''))
			os.unlink(file_path)
			
		print('Processing...')
		
		images_file_path	= os.path.join(self.root, self.raw_folder, 'images/')
		tr_lists_file_path	= os.path.join(self.root, self.raw_folder, 'lists/train.txt')
		te_lists_file_path	= os.path.join(self.root, self.raw_folder, 'lists/test.txt')
		
		train_files		= np.genfromtxt(tr_lists_file_path, dtype=str)
		test_files		= np.genfromtxt(te_lists_file_path, dtype=str)		
		train_data		= []
		test_data		= []
		train_labels		= []
		test_labels		= []
		for name in train_files:
			pathway	= os.path.join(images_file_path, name)
			img	= Image.open(pathway)
			img	= img.resize((64, 64), Image.ANTIALIAS)
			npimg	= np.array(img.getdata()).astype(float)
			npimg	= np.reshape(npimg, (img.size[0], img.size[1], 3))
			npimg	= np.transpose(npimg, (2, 0, 1))
			train_data.append(npimg)
			train_labels.append(int(name[0:3]) - 1)
			img.close()
			
		for name in test_files:
			pathway	= os.path.join(images_file_path, name)
			img	= Image.open(pathway)
			img	= img.resize((64, 64), Image.ANTIALIAS)
			npimg	= np.array(img.getdata()).astype(float)
			npimg	= np.reshape(npimg, (img.size[0], img.size[1], 3))
			npimg	= np.transpose(npimg, (2, 0, 1))
			test_data.append(npimg)
			test_labels.append(int(name[0:3]) - 1)
			img.close()

		train_data	= np.array(train_data)/255
		train_labels	= np.array(train_labels)
		test_data	= np.array(test_data)/255
		test_labels	= np.array(test_labels)
		
		assert train_data.shape[0] == 3000 and test_data.shape[0] == 3033
		assert train_labels.shape[0] == 3000 and test_labels.shape[0] == 3033
		
		training_set	= (
					torch.from_numpy(train_data).type(torch.FloatTensor),
					torch.from_numpy(train_labels).type(torch.LongTensor),
				  )
		testing_set	= (
					torch.from_numpy(test_data).type(torch.FloatTensor),
					torch.from_numpy(test_labels).type(torch.LongTensor),
				  )

		torch.save(training_set, os.path.join(self.root, self.processed_folder, self.training_file))
		torch.save(testing_set, os.path.join(self.root, self.processed_folder, self.test_file))
			
		print('Done!')
