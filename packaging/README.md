# Building torchvision packages for release

## Anaconda packages

### Linux

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/conda-cuda bash
pushd remote/conda

./build_vision.sh 9.0
./build_vision.sh 10.0
./build_vision.sh cpu

# copy packages over to /remote
# exit docker
# anaconda upload -u pytorch torchvision*.bz2
```

### OSX

```bash
# create a fresh anaconda environment / install and activate it
conda install -y conda-build anaconda-client
./build_vision.sh cpu

# copy packages over to /remote
# exit docker
# anaconda upload -u pytorch torchvision*.bz2
```

### Windows

```bash
# Open `Git Bash` and change dir to `conda`
./build_vision.sh 9.0
./build_vision.sh 10.0
./build_vision.sh cpu

# copy packages to a output directory
# anaconda upload -u pytorch torchvision*.bz2
```

## Wheels

### Linux

pushd wheel

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda90:latest bash
cd remote
./linux_manywheel.sh cu90

rm -rf /usr/local/cuda*
./linux_manywheel.sh cpu
```

```bash
nvidia-docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda100:latest bash
cd remote
./linux_manywheel.sh cu100
```

wheels are in the folders `cpu`, `cu90`, `cu100`.

You can upload the `cu90` wheels to twine with `twine upload *.whl`.
Which wheels we upload depends on which wheels PyTorch uploads as default, and right now, it's `cu90`.

### OSX

```bash
pushd wheel
./osx_wheel.sh
```

### Windows

```cmd
set PYTORCH_REPO=pytorch

pushd windows
call build_vision.bat 90 0.3.0 1
call build_vision.bat 100 0.3.0 1
call build_vision.bat cpu 0.3.0 1
```

wheels are in the current folder.

You can upload them to twine with `twine upload *.whl`
