#!/bin/bash

echo "Building vision dependencies and wheel started."

# Set environment variables
export SRC_PATH="$GITHUB_WORKSPACE/$SRC_DIR"
export CMAKE_BUILD_TYPE="$BUILD_TYPE"
export VCVARSALL_PATH="$DEPENDENCIES_DIR/VSBuildTools/VC/Auxiliary/Build/vcvarsall.bat"
export TRIPLET_FILE="triplets/arm64-windows.cmake"
export PYTORCH_VERSION="$PYTORCH_VERSION"
export CHANNEL="$CHANNEL"

# Dependencies
mkdir -p "$DOWNLOADS_DIR"
mkdir -p "$DEPENDENCIES_DIR"
echo "*" > "$DOWNLOADS_DIR/.gitignore"
echo "*" > "$DEPENDENCIES_DIR/.gitignore"

# Install vcpkg
cd "$DOWNLOADS_DIR" || exit
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg || exit
./bootstrap-vcpkg.sh

# Set vcpkg to only build release packages
echo "set(VCPKG_BUILD_TYPE release)" >> "$TRIPLET_FILE"

# Install dependencies using vcpkg
./vcpkg install libjpeg-turbo:arm64-windows --x-install-root="$DEPENDENCIES_DIR"
./vcpkg install libwebp:arm64-windows --x-install-root="$DEPENDENCIES_DIR"
./vcpkg install libpng[tools]:arm64-windows --x-install-root="$DEPENDENCIES_DIR"

# Copy files using cp
cp "$DEPENDENCIES_DIR/arm64-windows/lib/libpng16.lib" "$DEPENDENCIES_DIR/arm64-windows/lib/libpng.lib"
cp "$DEPENDENCIES_DIR/arm64-windows/bin/libpng16.dll" "$DEPENDENCIES_DIR/arm64-windows/bin/libpng.dll"
cp "$DEPENDENCIES_DIR/arm64-windows/bin/libpng16.pdb" "$DEPENDENCIES_DIR/arm64-windows/bin/libpng.pdb"
mkdir -p "$DEPENDENCIES_DIR/Library/"
cp -r "$DEPENDENCIES_DIR/arm64-windows/"* "$DEPENDENCIES_DIR/Library/"
cp -r "$DEPENDENCIES_DIR/Library/tools/libpng/"* "$DEPENDENCIES_DIR/Library/bin/"
cp -r "$DEPENDENCIES_DIR/Library/bin/"* "$SRC_PATH/torchvision"

# Source directory
cd "$SRC_PATH" || exit

# Create virtual environment
python -m pip install --upgrade pip
python -m venv .venv
echo "*" > .venv/.gitignore
source .venv/Scripts/activate

# Install dependencies
pip install numpy==2.2.3

if [ "$CHANNEL" = "release" ]; then
  echo "Installing latest stable version of PyTorch."
  # TODO: update when arm64 torch available on pypi
  pip3 install --pre torch --index-url https://download.pytorch.org/whl/torch/
elif [ "$CHANNEL" = "test" ]; then
  echo "Installing PyTorch version $PYTORCH_VERSION."
  pip3 install --pre torch=="$PYTORCH_VERSION" --index-url https://download.pytorch.org/whl/test
else
  echo "CHANNEL is not set, installing PyTorch from nightly."
  pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
fi

echo "Dependencies install finished successfully."
