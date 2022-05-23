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
GRADLE_PROPERTIES=/home/circleci/project/android/gradle.properties

echo "sdk.dir=${ANDROID_HOME}" >> $GRADLE_LOCAL_PROPERTIES
echo "ndk.dir=${ANDROID_NDK_HOME}" >> $GRADLE_LOCAL_PROPERTIES

echo "SONATYPE_NEXUS_USERNAME=${SONATYPE_NEXUS_USERNAME}" >> $GRADLE_PROPERTIES
echo "mavenCentralRepositoryUsername=${SONATYPE_NEXUS_USERNAME}" >> $GRADLE_PROPERTIES
echo "SONATYPE_NEXUS_PASSWORD=${SONATYPE_NEXUS_PASSWORD}" >> $GRADLE_PROPERTIES
echo "mavenCentralRepositoryPassword=${SONATYPE_NEXUS_PASSWORD}" >> $GRADLE_PROPERTIES

echo "signing.keyId=${ANDROID_SIGN_KEY}" >> $GRADLE_PROPERTIES
echo "signing.password=${ANDROID_SIGN_PASS}" >> $GRADLE_PROPERTIES

cat /home/circleci/project/android/gradle.properties | grep VERSION

${GRADLE_PATH} --scan --stacktrace --debug --no-daemon -p ${VISION_ANDROID} ops:uploadArchives

mkdir -p ~/workspace/artifacts
find . -type f -name *aar -print | xargs tar cfvz ~/workspace/artifacts/artifacts-aars.tgz
