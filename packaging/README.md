# Building torchvision packages for release

TorchVision release packages are built by using `build_wheel.sh` and `build_conda.sh` for all permutations of
supported operating systems, compute platforms and python versions.

OS/Python/Compute matrix is defined in https://github.com/pytorch/vision/blob/main/.circleci/regenerate.py

# Building custom wheels

When running on custom hardware (eg. Jetson), you may want to build your own wheels.  If so,
you should make sure that your torchvision wheel has a dependency on the correct custom torch
version.  
For eg. the `Poetry dependency resolver <https://python-poetry.org/>`_ this seems to be a 
requirement to make torch installations work.  

Proper version tagging can be achieved with the following procedure (example here for +cu113 tag)

.. code:: bash
    
    pip install torch==1.11.0+cu113
    export PYTORCH_VERSION=1.11.0+cu113 # Used to derive Pytorch dependency
    export BUILD_VERSION==0.12.0+cu113 # Used to generate Torchvision tag
    python setup.py bdist_wheel-

The resulting wheel will have a dependency on torch==1.11.0+cu113