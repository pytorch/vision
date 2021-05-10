#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"

TEST_APP_PATH=$(dirname $(realpath $0))
cd ${TEST_APP_PATH}

PYTORCH_IOS_NIGHTLY_NAME=libtorch_ios_nightly_build.zip
VISION_IOS_NIGHTLY_NAME=libtorchvision_ops_ios_nightly_build.zip

echo "Downloading torch libs and vision libs..."
wget https://ossci-ios-build.s3.amazonaws.com/${PYTORCH_IOS_NIGHTLY_NAME}
wget https://ossci-ios-build.s3.amazonaws.com/${VISION_IOS_NIGHTLY_NAME}

mkdir -p ./library/torch
mkdir -p ./library/vision

echo "Unziping torch libs and vision libs..."
unzip -d ./library/torch ./${PYTORCH_IOS_NIGHTLY_NAME}
unzip -d ./library/vision ./${VISION_IOS_NIGHTLY_NAME}

cp ./library/vision/install/lib/*.a ./library/torch/install/lib
cp -r ./library/torch/install .

rm -rf ./library
rm -rf ./*.zip

echo "Generating the vision model..."
python ./make_assets.py

echo "Finished project setups."
