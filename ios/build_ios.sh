#!/bin/bash

set -ex -o pipefail
echo ""
echo "DIR: $(pwd)"
VISION_IOS_ROOT=$(dirname $(realpath $0))

if ! [ -n "${LIBTORCH_HEADER_ROOT:-}" ]; then
  echo "Missing parameter: LIBTORCH_HEADER_ROOT"
  exit 1
fi

if [ -n "${IOS_ARCH:-}" ]; then
  if [ "${IOS_ARCH:-}" == "arm64" ]; then
    IOS_PLATFORM="OS"
  elif [ "${IOS_ARCH:-}" == "x86_64" ]; then
    IOS_PLATFORM="SIMULATOR"
  fi
fi

mkdir -p ${VISION_IOS_ROOT}/lib
mkdir -p ${VISION_IOS_ROOT}/build
cd ${VISION_IOS_ROOT}/build
cmake -DLIBTORCH_HEADER_ROOT=${LIBTORCH_HEADER_ROOT}  \
      -DCMAKE_TOOLCHAIN_FILE=${VISION_IOS_ROOT}/../cmake/iOS.cmake \
      -DIOS_ARCH=${IOS_ARCH} \
      -DIOS_PLATFORM=${IOS_PLATFORM} \
      ..
make
rm -rf ${VISION_IOS_ROOT}/build
