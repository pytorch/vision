import torch
from torch.utils.benchmark import Timer
from torchvision.io.image import decode_jpeg, read_file, ImageReadMode, write_jpeg, encode_jpeg
from torchvision import transforms as T

img_path = 'big_2kx2k.jpg'
img_path = 'test/assets/encode_jpeg/grace_hopper_517x606.jpg'
data = read_file(img_path)
batch_size = 32
batch_data = torch.cat([data] * batch_size, dim=0)
img = decode_jpeg(data)

def sumup(name, mean, median, throughput, fps):
    print(
        f"{name:<10} mean: {mean:.3f} ms, median: {median:.3f} ms, "
        f"Throughput = {throughput:.3f} Megapixel / sec, "
        f"{fps:.3f} fps"
    )

print(f"{img.shape = }")
print(f"{data.shape = }")
height, width = img.shape[-2:]

num_pixels = height * width
num_runs = 30

stmt = "decode_jpeg(data, device='{}')"
setup = 'from torchvision.io.image import decode_jpeg'
globals = {'data': data}

for device in ('cpu', 'cuda'):
    t = Timer(stmt=stmt.format(device), setup=setup, globals=globals).timeit(num_runs)
    sumup(device, t.mean * 1000, t.median * 1000, num_pixels / 1e6 / t.median, 1 / t.median)

# Benchmark batch
stmt = "torch.ops.image.decode_jpeg_batch_cuda(batch_data, mode, device, batch_size, height, width)"
setup = 'import torch'
globals = {
    'batch_data': batch_data, 'mode': ImageReadMode.UNCHANGED.value, 'device': torch.device('cuda'), 'batch_size': batch_size,
    'height': height, 'width': width
}
t = Timer(stmt=stmt, setup=setup, globals=globals).timeit(num_runs)

sumup("BATCH cuda", t.mean * 1000, t.median * 1000, num_pixels * batch_size / 1e6 / t.median, batch_size / t.median)

out = torch.ops.image.decode_jpeg_batch_cuda(batch_data, ImageReadMode.UNCHANGED.value, torch.device('cuda'), batch_size, height, width)
write_jpeg(out[0].to('cpu'), 'saved_imgs/first.jpg')