#!/bin/bash

VISION_IOS_ROOT=$(dirname $(realpath $0))

# Simple parsing of pod url
echo "Downloading LibTorch"
wget `pod search --no-pager --simple LibTorch | grep Source | head -1 | awk -v url=3 '{print $url}'` -O ${VISION_IOS_ROOT}/torch.zip
unzip ${VISION_IOS_ROOT}/torch.zip -d ${VISION_IOS_ROOT}/torch
rm ${VISION_IOS_ROOT}/torch.zip

echo "Building for x86_64"
LIBTORCH_HEADER_ROOT=${VISION_IOS_ROOT}/torch/install/include IOS_ARCH='x86_64' ${VISION_IOS_ROOT}/build_ios.sh
mv ${VISION_IOS_ROOT}/lib/libtorchvision_ops.a ${VISION_IOS_ROOT}/lib/libtorchvision_ops_x86_64.a

echo "Building for arm64"
LIBTORCH_HEADER_ROOT=${VISION_IOS_ROOT}/torch/install/include IOS_ARCH='arm64' ${VISION_IOS_ROOT}/build_ios.sh
mv ${VISION_IOS_ROOT}/lib/libtorchvision_ops.a ${VISION_IOS_ROOT}/lib/libtorchvision_ops_arm64.a

echo "Combining libs to fat binary"
lipo -create ${VISION_IOS_ROOT}/lib/libtorchvision_ops_arm64.a ${VISION_IOS_ROOT}/lib/libtorchvision_ops_x86_64.a -output ${VISION_IOS_ROOT}/lib/libtorchvision_ops.a
rm ${VISION_IOS_ROOT}/lib/libtorchvision_ops_arm64.a ${VISION_IOS_ROOT}/lib/libtorchvision_ops_x86_64.a


echo "Creating release"
cp ${VISION_IOS_ROOT}/../LICENSE ./
cd ${VISION_IOS_ROOT} && zip -r ./libtorchvision_ops_ios.zip LICENSE ./lib/
rm ${VISION_IOS_ROOT}/LICENSE


