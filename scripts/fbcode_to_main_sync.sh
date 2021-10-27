#!/bin/bash

if [ -z $1 ]
then
    echo "commit hash is required to be passed when running this script"
fi
git checkout fbsync
commit_hash=$1
IFS='
'
for line in $(git log --pretty=oneline "$commit_hash"..HEAD)
do
    if [[ $line != *\[fbsync\]* ]]
    then
        echo "Parsing $line"
        hash=$(echo $line | cut -f1 -d' ')
        git checkout master
        git checkout -B cherrypick_$hash
        git cherry-pick -x "$hash"
        git push origin cherrypick_$hash
        git checkout fbsync
    fi
done
echo "Please review the PRs, add [FBCode->GH] tag on the title and publish them."
