import unittest

import tempfile
import os
import shutil
import zipfile
from common_utils import get_tmp_dir

from torchvision.datasets import ZippedImageFolder
from torch._utils_internal import get_file_path_2


class Tester(unittest.TestCase):
    FAKEDATA_DIR = get_file_path_2(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fakedata')

    def test_zipped_image_folder(self):
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, "dataset.zip")
        try:
            with get_tmp_dir(src=os.path.join(Tester.FAKEDATA_DIR, 'imagefolder')) as root:
                classes = sorted(['a', 'b'])
                class_a_image_files = [os.path.join(root, 'a', file)
                                       for file in ('a1.png', 'a2.png', 'a3.png')]
                class_b_image_files = [os.path.join(root, 'b', file)
                                       for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')]

                zf = zipfile.ZipFile(temp_filename, "w", zipfile.ZIP_STORED, allowZip64=True)
                for dirname, subdirs, files in os.walk(root):
                    for filename in files:
                        zf.write(os.path.join(dirname, filename),
                                 os.path.relpath(os.path.join(dirname, filename), root))
                zf.close()

                dataset = ZippedImageFolder(root=temp_filename)
                for cls in classes:
                    self.assertEqual(cls, dataset.classes[dataset.class_to_idx[cls]])
                class_a_idx = dataset.class_to_idx['a']
                class_b_idx = dataset.class_to_idx['b']
                imgs_a = [(img_path.replace(root + os.path.sep, '').replace(os.path.sep, "/"), class_a_idx)
                          for img_path in class_a_image_files]
                imgs_b = [(img_path.replace(root + os.path.sep, '').replace(os.path.sep, "/"), class_b_idx)
                          for img_path in class_b_image_files]
                imgs = sorted(imgs_a + imgs_b)
                self.assertEqual(imgs, dataset.imgs)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
