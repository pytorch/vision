import os
import random

import torch
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t
from torchvision.io import VideoReader


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)

class KineticsRandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_multiplier=5, frame_transform=None, video_transform=None, clip_len=16, from_keyframes=True, alpha=0.1, device="cpu"):
        super(KineticsRandomDataset).__init__()

        self.samples = get_samples(root)
        self.epoch_size = epoch_multiplier * len(self.samples)
        self.clip_len = clip_len  # length of a clip in frames
        self.frame_transform = frame_transform  # transform for every frame individually
        self.video_transform = video_transform # transform on a video sequence
        # FIXME: maybe remove
        self.alpha = alpha # tollerance to avoid rounding errros with max seek time
        self.from_keyframes = from_keyframes  # if true, only decode from the keyframes
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for i in range(self.epoch_size):
            # get random sample
            path, target = random.choice(self.samples)
            # get video object
            vid = VideoReader(path, device=self.device)
            video_frames = [] # video frame buffer
            # seek and return frames
            metadata = vid.get_metadata()['video']
            duration = metadata['duration'][0] if self.device == 'cpu' else metadata['duration']
            fps = metadata['fps'][0] if self.device == 'cpu' else metadata['fps']
            max_seek = duration - (self.clip_len / fps + self.alpha)
            start = random.uniform(0., max_seek)
            vid.seek(start, keyframes_only=self.from_keyframes)
            while len(video_frames) < self.clip_len:
                frame = next(vid)['data']
                video_frames.append(self.frame_transform(frame) if self.frame_transform else frame)
            # stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
            }
            yield output



class KineticsSequentialDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_multiplier=5, frame_transform=None, video_transform=None, clip_len=16, alpha=0.1, device="cpu"):
        super(KineticsSequentialDataset).__init__()

        self.samples = get_samples(root)
        self.num_steps = epoch_multiplier // len(self.samples)
        self.clip_len = clip_len  # length of a clip in frames
        self.frame_transform = frame_transform  # transform for every frame individually
        self.video_transform = video_transform # transform on a video sequence
        # FIXME: maybe remove
        self.alpha = alpha # tollerance to avoid rounding errros with max seek time
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for i in range(len(self.samples)):
            # get random sample
            path, target = self.samples[i]
            # get video object
            vid = VideoReader(path, device=self.device)
            video_frames = [] # video frame buffer
            # seek and return frames

            metadata = vid.get_metadata()['video']
            duration = metadata['duration'][0] if self.device == 'cpu' else metadata['duration']
            fps = metadata['fps'][0] if self.device == 'cpu' else metadata['fps']
            max_seek = duration - (self.clip_len / fps + self.alpha)
            step = max(max_seek // self.num_steps, 1)
            tss = [i.item() for i in list(torch.linspace(0, max_seek, steps=self.num_steps))]
            for start in tss:
                vid.seek(start, keyframes_only=True)
                while len(video_frames) < self.clip_len:
                    frame = next(vid)['data']
                    video_frames.append(self.frame_transform(frame) if self.frame_transform else frame)
                # stack it into a tensor
                video = torch.stack(video_frames, 0)
                if self.video_transform:
                    video = self.video_transform(video)
                output = {
                    'path': path,
                    'video': video,
                    'target': target,
                    'start': start,
                }
                yield output
