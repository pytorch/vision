#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"

WORKSPACE=/Users/distiller/workspace
PROJ_ROOT=/Users/distiller/project
ARTIFACTS_DIR=${WORKSPACE}/ios
ls ${ARTIFACTS_DIR}


# Install conda then 'conda install' awscli
curl --retry 3 -o ~/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ~/conda.sh
/bin/bash ~/conda.sh -b -p ~/anaconda
export PATH="~/anaconda/bin:${PATH}"
source ~/anaconda/bin/activate
conda install wget --yes
conda install -c conda-forge awscli --yes

DOWNLOADS_DIR=${WORKSPACE}/downloads
mkdir -p ${DOWNLOADS_DIR}
cd ${DOWNLOADS_DIR}

PYTORCH_IOS_NIGHTLY_NAME=libtorch_ios_nightly_build.zip
wget https://ossci-ios-build.s3.amazonaws.com/${PYTORCH_IOS_NIGHTLY_NAME}
unzip -d . ./${PYTORCH_IOS_NIGHTLY_NAME}


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
# ZIPFILE=libtorchvision_ops_ios_nightly_build.zip
# cd ${ZIP_DIR}
# #for testing
# touch version.txt
# echo $(date +%s) > version.txt
# zip -r ${ZIPFILE} install version.txt LICENSE

ZIPFILE=libtorchvision_ops_ios_nightly_build_with_torch.zip
cp ${ZIP_DIR}/install/lib/*.a  ${DOWNLOADS_DIR}/install/lib
cd ${DOWNLOADS_DIR}
zip -r ${ZIPFILE} install

# upload to aws

set +x
export AWS_ACCESS_KEY_ID=${AWS_S3_ACCESS_KEY_FOR_PYTORCH_BINARY_UPLOAD}
export AWS_SECRET_ACCESS_KEY=${AWS_S3_ACCESS_SECRET_FOR_PYTORCH_BINARY_UPLOAD}
set -x
aws s3 cp ${ZIPFILE} s3://ossci-ios-build/ --acl public-read
