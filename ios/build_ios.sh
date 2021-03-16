#!/bin/bash

set -ex -o pipefail
echo ""
echo "DIR: $(pwd)"
VISION_IOS_ROOT=$(dirname $(realpath $0))

if ! [ -n "${LIBTORCH_HEADER_ROOT:-}" ]; then
  echo "Missing parameter: LIBTORCH_HEADER_ROOT"
  exit 1
fi

mkdir -p ${VISION_IOS_ROOT}/lib
mkdir -p ${VISION_IOS_ROOT}/build
cd ${VISION_IOS_ROOT}/build
cmake -DLIBTORCH_HEADER_ROOT=${LIBTORCH_HEADER_ROOT} ..
make
rm -rf ${VISION_IOS_ROOT}/build
