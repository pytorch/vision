#!/bin/bash
set -ex -o pipefail

TEST_APP_PATH=$(dirname $(realpath $0))
cd ${TEST_APP_PATH}

rm -rf ./install
rm ./VisionTestApp/*.pt
