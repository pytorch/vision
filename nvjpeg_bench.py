import torch
from torch.utils.benchmark import Timer
from torchvision.io.image import decode_jpeg, read_file, ImageReadMode, write_jpeg, encode_jpeg
from torchvision import transforms as T
import sys

img_path = sys.argv[1]
data = read_file(img_path)
img = decode_jpeg(data)
write_jpeg(T.Resize((300, 300))(img), 'lol.jpg')


def sumup(name, mean, median, throughput, fps):
    print(
        f"{name:<20} - mean: {mean:<7.2f} ms, median: {median:<7.2f} ms, "
        f"Throughput = {throughput:<7.1f} Megapixel / sec, "
        f"{fps:<7.1f} fps"
    )

print(f"Using {img_path}")
print(f"{img.shape = }, {data.shape = }")
height, width = img.shape[-2:]

num_pixels = height * width
num_runs = 30


for batch_size in (1, 4, 16, 32, 64):
    print(f"{batch_size = }")

    # non-batch implem
    for device in ('cpu', 'cuda'):
        if batch_size >= 32 and height >= 1000 and device == 'cuda':
            print(f"skipping for-loop for {batch_size = } and {device = }")
            continue
        stmt = f"for _ in range(batch_size): decode_jpeg(data, device='{device}')"
        setup = 'from torchvision.io.image import decode_jpeg'
        globals = {'data': data, 'batch_size': batch_size}

        t = Timer(stmt=stmt, setup=setup, globals=globals).timeit(num_runs)
        sumup(f"for-loop {device}", t.mean * 1000, t.median * 1000, num_pixels * batch_size / 1e6 / t.median, batch_size / t.median)

    # # Batch implem
    # stmt = "torch.ops.image.decode_jpeg_batch_cuda(batch_data, mode, device, batch_size, height, width)"
    # setup = 'import torch'
    # batch_data = torch.cat([data] * batch_size, dim=0)
    # globals = {
    #     'batch_data': batch_data, 'mode': ImageReadMode.UNCHANGED.value, 'device': torch.device('cuda'), 'batch_size': batch_size,
    #     'height': height, 'width': width
    # }
    # t = Timer(stmt=stmt, setup=setup, globals=globals).timeit(num_runs)

    # sumup(f"BATCH cuda", t.mean * 1000, t.median * 1000, num_pixels * batch_size / 1e6 / t.median, batch_size / t.median)

