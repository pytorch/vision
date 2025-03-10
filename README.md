# torchvision

[![total torchvision downloads](https://pepy.tech/badge/torchvision)](https://pepy.tech/project/torchvision)
[![documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/vision/stable/index.html)

The torchvision package consists of popular datasets, model architectures, and common image transformations for computer
vision.

## Installation

Please refer to the [official
instructions](https://pytorch.org/get-started/locally/) to install the stable
versions of `torch` and `torchvision` on your system.

To build source, refer to our [contributing
page](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation).

The following is the corresponding `torchvision` versions and supported Python
versions.

| `torch`            | `torchvision`      | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.9`, `<=3.12`   |
| `2.5`              | `0.20`             | `>=3.9`, `<=3.12`   |
| `2.4`              | `0.19`             | `>=3.8`, `<=3.12`   |
| `2.3`              | `0.18`             | `>=3.8`, `<=3.12`   |
| `2.2`              | `0.17`             | `>=3.8`, `<=3.11`   |
| `2.1`              | `0.16`             | `>=3.8`, `<=3.11`   |
| `2.0`              | `0.15`             | `>=3.8`, `<=3.11`   |

<details>
    <summary>older versions</summary>

| `torch` | `torchvision`     | Python                    |
|---------|-------------------|---------------------------|
| `1.13`  | `0.14`            | `>=3.7.2`, `<=3.10`       |
| `1.12`  | `0.13`            | `>=3.7`, `<=3.10`         |
| `1.11`  | `0.12`            | `>=3.7`, `<=3.10`         |
| `1.10`  | `0.11`            | `>=3.6`, `<=3.9`          |
| `1.9`   | `0.10`            | `>=3.6`, `<=3.9`          |
| `1.8`   | `0.9`             | `>=3.6`, `<=3.9`          |
| `1.7`   | `0.8`             | `>=3.6`, `<=3.9`          |
| `1.6`   | `0.7`             | `>=3.6`, `<=3.8`          |
| `1.5`   | `0.6`             | `>=3.5`, `<=3.8`          |
| `1.4`   | `0.5`             | `==2.7`, `>=3.5`, `<=3.8` |
| `1.3`   | `0.4.2` / `0.4.3` | `==2.7`, `>=3.5`, `<=3.7` |
| `1.2`   | `0.4.1`           | `==2.7`, `>=3.5`, `<=3.7` |
| `1.1`   | `0.3`             | `==2.7`, `>=3.5`, `<=3.7` |
| `<=1.0` | `0.2`             | `==2.7`, `>=3.5`, `<=3.7` |

</details>

## Image Backends

Torchvision currently supports the following image backends:

- torch tensors
- PIL images:
    - [Pillow](https://python-pillow.org/)
    - [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) - a **much faster** drop-in replacement for Pillow with SIMD.

Read more in in our [docs](https://pytorch.org/vision/stable/transforms.html).

## [UNSTABLE] Video Backend

Torchvision currently supports the following video backends:

- [pyav](https://github.com/PyAV-Org/PyAV) (default) - Pythonic binding for ffmpeg libraries.
- video_reader - This needs ffmpeg to be installed and torchvision to be built from source. There shouldn't be any
  conflicting version of ffmpeg installed. Currently, this is only supported on Linux.

```
conda install -c conda-forge 'ffmpeg<4.3'
python setup.py install
```

# Using the models on C++

Refer to [example/cpp](https://github.com/pytorch/vision/tree/main/examples/cpp).

**DISCLAIMER**: the `libtorchvision` library includes the torchvision
custom ops as well as most of the C++ torchvision APIs. Those APIs do not come
with any backward-compatibility guarantees and may change from one version to
the next. Only the Python APIs are stable and with backward-compatibility
guarantees. So, if you need stability within a C++ environment, your best bet is
to export the Python APIs via torchscript.

## Documentation

You can find the API documentation on the pytorch website: <https://pytorch.org/vision/stable/index.html>

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## Disclaimer on Datasets

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets,
vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to
determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset
to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML
community!

## Pre-trained Model License

The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the
dataset used for training. It is your responsibility to determine whether you have permission to use the models for your
use case.

More specifically, SWAG models are released under the CC-BY-NC 4.0 license. See
[SWAG LICENSE](https://github.com/facebookresearch/SWAG/blob/main/LICENSE) for additional details.

## Citing TorchVision

If you find TorchVision useful in your work, please consider citing the following BibTeX entry:

```bibtex
@software{torchvision2016,
    title        = {TorchVision: PyTorch's Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
```
