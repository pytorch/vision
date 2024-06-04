#!/usr/bin/env bash
#
# Usage (run from root of project):
# TEST_INFRA_BRANCH=release/2.1 RELEASE_BRANCH=release/2.1 RELEASE_VERSION=2.1.0 packaging/cut_release.sh
#
# TEST_INFRA_BRANCH: The release branch of test-infra that houses all reusable
# workflows
#
# RELEASE_BRANCH: The name of the release branch for this repo
#
# RELEASE_VERSION: Version of this current release

set -eou pipefail

# Create and Check out to Release Branch
git checkout -b "${RELEASE_BRANCH}"

# Change all GitHub Actions to reference the test-infra release branch
# as opposed to main.
for i in .github/workflows/*.yml; do 
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e s#@main#@"${TEST_INFRA_BRANCH}"# $i;
    sed -i '' -e s#test-infra-ref:[[:space:]]main#"test-infra-ref: ${TEST_INFRA_BRANCH}"# $i;
  else
    sed -i -e s#@main#@"${TEST_INFRA_BRANCH}"# $i;
    sed -i -e s#test-infra-ref:[[:space:]]main#"test-infra-ref: ${TEST_INFRA_BRANCH}"# $i;
  fi
done

# Update the Release Version in version.txt
echo "${RELEASE_VERSION}" >version.txt

# Optional
# git add ./github/workflows/*.yml version.txt
# git commit -m "[RELEASE-ONLY CHANGES] Branch Cut for Release {RELEASE_VERSION}"
# git push origin "${RELEASE_BRANCH}"
