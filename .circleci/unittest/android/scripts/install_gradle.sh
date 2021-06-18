#!/bin/bash
set -ex

_https_amazon_aws=https://ossci-android.s3.amazonaws.com
GRADLE_VERSION=6.8.3

_gradle_home=/opt/gradle
sudo rm -rf $gradle_home
sudo mkdir -p $_gradle_home

curl --silent --output /tmp/gradle.zip --retry 3 $_https_amazon_aws/gradle-${GRADLE_VERSION}-bin.zip

sudo unzip -q /tmp/gradle.zip -d $_gradle_home
rm /tmp/gradle.zip

sudo chmod -R 777 $_gradle_home

export GRADLE_HOME=$_gradle_home/gradle-$GRADLE_VERSION
export GRADLE_PATH=${GRADLE_HOME}/bin/gradle
