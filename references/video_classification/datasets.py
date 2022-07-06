import torchvision
from typing import Tuple
from torch import Tensor

class CustomKinetics(torchvision.datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, video_idx, audio, label
