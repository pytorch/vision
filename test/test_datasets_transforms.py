import os
import shutil
import contextlib
import tempfile
import unittest
from torchvision.datasets import ImageFolder

FAKEDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'assets', 'fakedata')


@contextlib.contextmanager
def tmp_dir(src=None, **kwargs):
    tmp_dir = tempfile.mkdtemp(**kwargs)
    if src is not None:
        os.rmdir(tmp_dir)
        shutil.copytree(src, tmp_dir)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


def mock_transform(return_value, arg_list):
    def mock(arg):
        arg_list.append(arg)
        return return_value
    return mock


class Tester(unittest.TestCase):
    def test_transform(self):
        with tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
            class_a_image_files = [os.path.join(root, 'a', file)
                                   for file in ('a1.png', 'a2.png', 'a3.png')]
            class_b_image_files = [os.path.join(root, 'b', file)
                                   for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')]
            return_value = os.path.join(root, 'a', 'a1.png')
            args = []
            transform = mock_transform(return_value, args)
            dataset = ImageFolder(root, loader=lambda x: x, transform=transform)

            outputs = [dataset[i][0] for i in range(len(dataset))]
            self.assertEqual([return_value] * len(outputs), outputs)

            imgs = sorted(class_a_image_files + class_b_image_files)
            self.assertEqual(imgs, sorted(args))

    def test_target_transform(self):
        with tmp_dir(src=os.path.join(FAKEDATA_DIR, 'imagefolder')) as root:
            class_a_image_files = [os.path.join(root, 'a', file)
                                   for file in ('a1.png', 'a2.png', 'a3.png')]
            class_b_image_files = [os.path.join(root, 'b', file)
                                   for file in ('b1.png', 'b2.png', 'b3.png', 'b4.png')]
            return_value = os.path.join(root, 'a', 'a1.png')
            args = []
            target_transform = mock_transform(return_value, args)
            dataset = ImageFolder(root, loader=lambda x: x,
                                  target_transform=target_transform)

            outputs = [dataset[i][1] for i in range(len(dataset))]
            self.assertEqual([return_value] * len(outputs), outputs)

            class_a_idx = dataset.class_to_idx['a']
            class_b_idx = dataset.class_to_idx['b']
            targets = sorted([class_a_idx] * len(class_a_image_files) +
                             [class_b_idx] * len(class_b_image_files))
            self.assertEqual(targets, sorted(args))


if __name__ == '__main__':
    unittest.main()
