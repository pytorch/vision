from .video_utils import VideoClips
from .utils import list_dir
from .folder import make_dataset
from .vision import VisionDataset


class KineticsVideo(VisionDataset):
    def __init__(self, root, frames_per_clip, step_between_clips=1):
        super(KineticsVideo, self).__init__(root)
        extensions = ('avi',)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        return video, audio, label
