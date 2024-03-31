Hello World!
============

This is a minimal example of getting TorchVision to work in C++ with CMake.


In order to successfully compile this example, make sure you have both ``LibTorch`` and
``TorchVision`` installed.
Once both dependencies are sorted, we can start the CMake fun:

1) Create a ``build`` directory inside the current one.
2) from within the ``build`` directory, run the following commands:
    - ``python ../trace_model.py`` To use a torchvision model in C++, you must first export it from the python version of torchvision. More information can be found on the corresponding `documentation page <https://pytorch.org/tutorials/advanced/cpp_export.html#loading-a-torchscript-model-in-c>`_.
    - | ``cmake -DCMAKE_PREFIX_PATH="<PATH_TO_LIBTORCH>;<PATH_TO_TORCHVISION>" ..``
      | where ``<PATH_TO_LIBTORCH>`` and ``<PATH_TO_TORCHVISION>`` are the paths to the libtorch and torchvision installations.
    - ``cmake --build .``

| That's it!
| You should now have a ``hello-world`` executable in your ``build`` folder.
 Running it will output a (fairly long) tensor of random values to your terminal.
