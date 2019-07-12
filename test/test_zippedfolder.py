import unittest

import tempfile
import os
import shutil
import zipfile

from torchvision.datasets import ZippedImageFolder
from torch._utils_internal import get_file_path_2


class Tester(unittest.TestCase):
    FAKEDATA_DIR = get_file_path_2(
            os.path.dirname(os.path.abspath(__file__)), 'assets', 'fakedata')
    root = os.path.normpath(FAKEDATA_DIR)
    classes = ['a', 'b']
    class_a_images = [os.path.normpath(get_file_path_2(os.path.join(root, 'imagefolder/a/', path)))
                      for path in ['a1.png', 'a2.png', 'a3.png']]
    class_b_images = [os.path.normpath(get_file_path_2(os.path.join(root, 'imagefolder/b/', path)))
                      for path in ['b1.png', 'b2.png', 'b3.png', 'b4.png']]

    def test_zipped_image_folder(self):
        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, "dataset.zip")
        try:
            zf = zipfile.ZipFile(temp_filename, "w", zipfile.ZIP_STORED, allowZip64=True)
            for dirname, subdirs, files in os.walk(Tester.root):
                for filename in files:
                    zf.write(os.path.join(dirname, filename),
                             os.path.relpath(os.path.join(dirname, filename), Tester.root))
            zf.close()

            dataset = ZippedImageFolder(root=temp_filename)
            for cls in Tester.classes:
                self.assertEqual(cls, dataset.classes[dataset.class_to_idx[cls]])
            class_a_idx = dataset.class_to_idx['a']
            class_b_idx = dataset.class_to_idx['b']
            imgs_a = [(img_path.replace(os.path.join(root, 'imagefolder/'), ''), class_a_idx) for img_path in Tester.class_a_images]
            imgs_b = [(img_path.replace(os.path.join(root, 'imagefolder/'), ''), class_b_idx) for img_path in Tester.class_b_images]
            imgs = sorted(imgs_a + imgs_b)
            self.assertEqual(imgs, dataset.imgs)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
