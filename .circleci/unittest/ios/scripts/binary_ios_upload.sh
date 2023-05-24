#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"

WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project
ARTIFACTS_DIR=${WORKSPACE}/ios
ls ${ARTIFACTS_DIR}
ZIP_DIR=${WORKSPACE}/zip
mkdir -p ${ZIP_DIR}/install/lib

# build a FAT bianry
cd ${ZIP_DIR}/install/lib
libs=("${ARTIFACTS_DIR}/x86_64/libtorchvision_ops.a" "${ARTIFACTS_DIR}/arm64/libtorchvision_ops.a")
lipo -create "${libs[@]}" -o ${ZIP_DIR}/install/lib/libtorchvision_ops.a
lipo -i ${ZIP_DIR}/install/lib/*.a

# copy the license
cp ${PROJ_ROOT}/LICENSE ${ZIP_DIR}/
# zip the library
ZIPFILE=libtorchvision_ops_ios_nightly_build.zip
cd ${ZIP_DIR}
#for testing
touch version.txt
echo $(date +%s) > version.txt
zip -r ${ZIPFILE} install version.txt LICENSE

# upload to aws
# Install conda then 'conda install' awscli
curl --retry 3 -o ~/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ~/conda.sh
/bin/bash ~/conda.sh -b -p ~/anaconda
export PATH="~/anaconda/bin:${PATH}"
source ~/anaconda/bin/activate
conda install -c conda-forge awscli --yes
set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set -x
aws s3 cp ${ZIPFILE} s3://ossci-ios-build/ --acl public-read
