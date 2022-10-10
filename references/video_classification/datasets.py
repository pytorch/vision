from typing import Tuple

import torch
from torchvision import datasets
from torchvision.prototype import features


class KineticsWithVideoId(datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, features.Label, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        video = features.Video(video)
        label = features.Label(label, categories=self.classes)

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx
