from torchvision.prototype import datasets
from torchvision.prototype.datasets.video_utils import AVKeyframeReader, AVRandomFrameReader, AVClipReader



print("\n \n KEYFRAMES \n \n")
ct = 0
dataset = AVKeyframeReader(datasets.load("ucf101"))
for i in dataset:
    print(i)
    ct += 1
    if ct > 5:
        break 


print("\n \n RANDOM FRAMES")
ct = 0
dataset = AVRandomFrameReader(datasets.load("ucf101"), num_samples=3)
for i in dataset:
    print(i)
    ct += 1
    if ct > 5:
        break

print("\n \n CLIPS ")
ct = 0
dataset = AVClipReader(datasets.load("ucf101"), num_frames_per_clip=16, num_clips_per_video=8)
for i in dataset:
    print(i['path'], i["range"])
    ct += 1
    if ct > 5:
        break  