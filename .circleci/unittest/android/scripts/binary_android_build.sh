#!/bin/bash
set -ex -o pipefail

echo "DIR: $(pwd)"
echo "ANDROID_HOME=${ANDROID_HOME}"
echo "ANDROID_NDK_HOME=${ANDROID_NDK_HOME}"
echo "JAVA_HOME=${JAVA_HOME}"

WORKSPACE=/home/circleci/workspace
VISION_ANDROID=/home/circleci/project/android

. /home/circleci/project/.circleci/unittest/android/scripts/install_gradle.sh

GRADLE_LOCAL_PROPERTIES=${VISION_ANDROID}/local.properties
rm -f $GRADLE_LOCAL_PROPERTIES

echo "sdk.dir=${ANDROID_HOME}" >> $GRADLE_LOCAL_PROPERTIES
echo "ndk.dir=${ANDROID_NDK_HOME}" >> $GRADLE_LOCAL_PROPERTIES

echo "GRADLE_PATH $GRADLE_PATH"
echo "GRADLE_HOME $GRADLE_HOME"

${GRADLE_PATH} --scan --stacktrace --debug --no-daemon -p ${VISION_ANDROID} assemble || true

mkdir -p ~/workspace/artifacts
find . -type f -name *aar -print | xargs tar cfvz ~/workspace/artifacts/artifacts-aars.tgz
find . -type f -name *apk -print | xargs tar cfvz ~/workspace/artifacts/artifacts-apks.tgz
