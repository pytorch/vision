from .vision import VisionDataset
from PIL import Image
import os
import os.path
import io
import string
from collections.abc import Iterable
import pickle
from .utils import verify_str_arg, iterable_to_str


class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = ("Expected type str for elements in argument classes, "
                          "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)
