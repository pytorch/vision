import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class LSUNClass(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + db_path.replace('/', '_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load( open( cache_file, "rb" ) )
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [ key for key, _ in txn.cursor() ]
            pickle.dump( self.keys, open( cache_file, "wb" ) )
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
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

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class LSUN(data.Dataset):
    """
    db_path = root directory for the database files
    classes = 'train' | 'val' | 'test' | ['bedroom_train', 'church_train', ...]
    """
    def __init__(self, db_path, classes='train',
                 transform=None, target_transform=None):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']
        self.db_path = db_path
        if type(classes) == str and classes in dset_opts:
            classes = [c + '_' + classes for c in categories]
        if type(classes) == list:
            for c in classes:
                c_short = c.split('_')
                c_short.pop(len(c_short) - 1)
                c_short = '_'.join(c_short)
                if c_short not in categories:
                    raise(ValueError('Unknown LSUN class: ' + c_short + '.'\
                          'Options are: ' + str(categories)))
                c_short = c.split('_')
                c_short = c_short.pop(len(c_short) - 1)
                if c_short not in dset_opts:
                    raise(ValueError('Unknown postfix: ' + c_short + '.'\
                          'Options are: ' + str(dset_opts)))
        else:
            raise(ValueError('Unknown option for classes'))
        self.classes = classes

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                db_path = db_path + '/' + c + '_lmdb',
                transform = transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count
        self.target_transform = target_transform

    def __getitem__(self, index):
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub += ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

if __name__ == '__main__':
    #lsun = LSUNClass(db_path='/home/soumith/local/lsun/train/bedroom_train_lmdb')
    #a = lsun[0]
    lsun = LSUN(db_path='/home/soumith/local/lsun/train',
                       classes=['bedroom_train', 'church_outdoor_train'])
    print(lsun.classes)
    print(lsun.dbs)
    a, t = lsun[len(lsun)-1]
    print(a)
    print(t)
