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
| `main` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11`   |
| `2.1`              | `0.16`             | `>=3.8`, `<=3.11`   |
| `2.0`              | `0.15`             | `>=3.8`, `<=3.11`   |
| `1.13`             | `0.14`             | `>=3.7.2`, `<=3.10` |

<details>
    <summary>older versions</summary>

| `torch` | `torchvision`     | Python                    |
|---------|-------------------|---------------------------|
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

TorchVision provides an example project for how to use the models on C++ using JIT Script.

Installation From source:

```
mkdir build
cd build
# Add -DWITH_CUDA=on support for the CUDA if needed
cmake ..
make
make install
```

Once installed, the library can be accessed in cmake (after properly configuring `CMAKE_PREFIX_PATH`) via the
`TorchVision::TorchVision` target:

```
find_package(TorchVision REQUIRED)
target_link_libraries(my-target PUBLIC TorchVision::TorchVision)
```

The `TorchVision` package will also automatically look for the `Torch` package and add it as a dependency to
`my-target`, so make sure that it is also available to cmake via the `CMAKE_PREFIX_PATH`.

For an example setup, take a look at `examples/cpp/hello_world`.

Python linking is disabled by default when compiling TorchVision with CMake, this allows you to run models without any
Python dependency. In some special cases where TorchVision's operators are used from Python code, you may need to link
to Python. This can be done by passing `-DUSE_PYTHON=on` to CMake.

### TorchVision Operators

In order to get the torchvision operators registered with torch (eg. for the JIT), all you need to do is to ensure that
you `#include <torchvision/vision.h>` in your project.

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
