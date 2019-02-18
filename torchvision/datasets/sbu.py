from PIL import Image
from six.moves import zip
from .utils import download_url, check_integrity

import os
from .vision import VisionDataset


class SBU(VisionDataset):
    """`SBU Captioned Photo <http://www.cs.virginia.edu/~vicente/sbucaptions/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where tarball
            ``SBUCaptionedPhotoDataset.tar.gz`` exists.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = "http://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz"
    filename = "SBUCaptionedPhotoDataset.tar.gz"
    md5_checksum = '9aec147b3488753cf758b4d493422285'

    def __init__(self, root, transform=None, target_transform=None,
                 download=True):
        super(SBU, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # Read the caption for each photo
        self.photos = []
        self.captions = []

        file1 = os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_urls.txt')
        file2 = os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_captions.txt')

        for line1, line2 in zip(open(file1), open(file2)):
            url = line1.rstrip()
            photo = os.path.basename(url)
            filename = os.path.join(self.root, 'dataset', photo)
            if os.path.exists(filename):
                caption = line2.rstrip()
                self.photos.append(photo)
                self.captions.append(caption)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a caption for the photo.
        """
        filename = os.path.join(self.root, 'dataset', self.photos[index])
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = self.captions[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """The number of photos in the dataset."""
        return len(self.photos)

    def _check_integrity(self):
        """Check the md5 checksum of the downloaded tarball."""
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        """Download and extract the tarball, and download each individual photo."""
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.md5_checksum)

        # Extract file
        with tarfile.open(os.path.join(self.root, self.filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

        # Download individual photos
        with open(os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_urls.txt')) as fh:
            for line in fh:
                url = line.rstrip()
                try:
                    download_url(url, os.path.join(self.root, 'dataset'))
                except OSError:
                    # The images point to public images on Flickr.
                    # Note: Images might be removed by users at anytime.
                    pass
