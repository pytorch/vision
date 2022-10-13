import torch
import tqdm
from torchvision.prototype.transforms import functional as F


torch.manual_seed(0)
video = torch.testing.make_tensor(4, 5, 3, 7, 33, low=0, high=255, dtype=torch.uint8, device="cpu")

num_calls = 1_000_000
num_failing = 0
for _ in tqdm.tqdm(range(num_calls)):
    output = F.gaussian_blur_video(video, kernel_size=3)
    if output[3, 0, 0, 2, 10] != 150:
        num_failing += 1

print(f"{num_failing:_} ({num_failing / num_calls:.1%}) calls exhibited non-determinism")
