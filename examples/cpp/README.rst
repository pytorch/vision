Using torchvision models in C++
===============================

This is a minimal example of getting TorchVision models to work in C++ with
Torchscript. The model is first scripted in Python and exported to a file, and
then loaded in C++. For a similar tutorial, see [this
tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html).

In order to successfully compile this example, make sure you have ``LibTorch``
installed. You can either:

- Install PyTorch normally
- Or download the LibTorch C++ distribution.

In both cases refer [here](https://pytorch.org/get-started/locally/) the
corresponding install or download instructions.

Some torchvision models only depend on PyTorch operators, and can be used in C++
without depending on the torchvision lib. Other models rely on torchvision's C++
operators like NMS, RoiAlign (typically the detection models) and those need to
be linked against the torchvision lib.

We'll first see the simpler case of running a model without the torchvision lib
dependency.

Running a model that doesn't need torchvision lib
-------------------------------------------------

Create a ``build`` directory inside the current one.

```bash
mkdir build
cd build
```

Then run `python ../trace_model.py` which should create a `resnet18.pt` file in
the build directory. This is the scripted model that will be used in the C++
code.

We can now start building with CMake. We have to tell CMake where it can find
the necessary PyTorch resources. If you installed PyTorch normally, you can do:

```bash
TORCH_PATH=$(python -c "import pathlib, torch; print(pathlib.Path(torch.__path__[0]))")
Torch_DIR="${TORCH_PATH}/share/cmake/Torch"   # there should be .cmake files in there

cmake .. -DTorch_DIR=$Torch_DIR
```

If instead you downloaded the LibTorch somewhere, you can do:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

Then `cmake --build .` and you should now be able to run

```bash
./run_model resnet18.pt
```

If you try to run the model with a model that depends on the torchvision lib, like
`./run_model fasterrcnn_resnet50_fpn.pt`, you should get a runtime error. This is
because the executable wasn't linked against the torchvision lib.


Running a model that needs torchvision lib
------------------------------------------

First, we need to build the torchvision lib. To build the torchvision lib go to
the root of the torchvision project and run:

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch  # or -DTorch_DIR= if you installed PyTorch normally, see above
cmake --build .
cmake --install .
```

You may want to pass `-DCMAKE_INSTALL_PREFIX=/path/to/libtorchvision` for
cmake to copy/install the files to a specific location (e.g. `$CONDA_PREFIX`).

**DISCLAIMER**: the `libtorchvision` library includes the torchvision
custom ops as well as most of the C++ torchvision APIs. Those APIs do not come
with any backward-compatibility guarantees and may change from one version to
the next. Only the Python APIs are stable and with backward-compatibility
guarantees. So, if you need stability within a C++ environment, your best bet is
to export the Python APIs via torchscript.

Now that libtorchvision is built and installed we can tell our project to use
and link to it via the `-DUSE_TORCHVISION` flag. We also need to tell CMake
where to find it, just like we did with LibTorch, e.g.:

```bash
cmake .. -DTorch_DIR=$Torch_DIR -DTorchVision_DIR=path/to/libtorchvision -DUSE_TORCHVISION=ON
cmake --build .
```

Now the `run_model` executable should be able to run the
`fasterrcnn_resnet50_fpn.pt` file.
