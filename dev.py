import torch, torchvision
video_path = "/home/bjuncek/work/video_reader_benchmark/videos/R6llTwEh07w.mp4"
video = torch.classes.torchvision.Video(video_path, "video:0", True)
video.next("video")