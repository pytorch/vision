import torchvision

import unittest
from unittest import mock
from unittest.mock import patch



def download_url(url, root, filename=None, md5=None):
    print("Downloaded {} to {} with filename {} and md5 {}".format(url, root, filename, md5))



class Tester(unittest.TestCase):
    #@mock.patch('torchvision.datasets.utils.download_url')
    @mock.patch('torchvision.datasets.mnist.download_url')
    @mock.patch('torchvision.datasets.mnist.MNIST._check_exists')
    @mock.patch('torchvision.datasets.mnist.read_image_file')
    @mock.patch('torchvision.datasets.mnist.read_label_file')
    def test_mnist(self, check_fn, download_fn):
        dataset = torchvision.datasets.MNIST('.')

        



if __name__ == '__main__':
    #unittest.main()
    with patch('torchvision.datasets.utils.download_url', side_effect=download_url) as f:
        torchvision.datasets.utils.download_url('http://google.com', '.')
