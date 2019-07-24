import glob
import os

from .video_utils import VideoClips
from .utils import list_dir
from .folder import make_dataset
from .vision import VisionDataset


class HMDB51(VisionDataset):

    data_url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    splits = {
        "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "md5": "15e67781e70dcfbdce2d7dbb9b3344b5"
    }

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 fold=1, train=True):
        super(HMDB51, self).__init__(root)
        extensions = ('avi',)
        self.fold = fold
        self.train = train

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(indices)

    def _select_fold(self, video_list, annotation_path, fold, train):
        target_tag = 1 if train else 2
        name = "*test_split{}.txt".format(fold)
        files = glob.glob(os.path.join(annotation_path, name))
        selected_files = []
        for f in files:
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                data = [x[0] for x in data if int(x[1]) == target_tag]
                selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if os.path.basename(video_list[i]) in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        return video, audio, label
