import os
import shutil
import tempfile
import torchvision.datasets as datasets
import unittest


class Tester(unittest.TestCase):

    def test_celeba(self):
        temp_dir = tempfile.mkdtemp()
        ds = datasets.CelebA(root=temp_dir, download=True)
        assert len(ds) == 162770
        assert ds[40711] is not None

        # 2nd time, the ZIP file will be detected (because now it has been downloaded)
        ds2 = datasets.CelebA(root=temp_dir, download=True)
        assert len(ds2) == 162770
        assert ds2[40711] is not None
        shutil.rmtree(temp_dir)

    def test_omniglot(self):
        temp_dir = tempfile.mkdtemp()
        ds = datasets.Omniglot(root=temp_dir, download=True)
        assert len(ds) == 19280
        assert ds[4071] is not None

        # 2nd time, the ZIP file will be detected (because now it has been downloaded)
        ds2 = datasets.Omniglot(root=temp_dir, download=True)
        assert len(ds2) == 19280
        assert ds2[4071] is not None
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
