#!/bin/bash

if [ -z $1 ]
then
    echo "Commit hash is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
commit_hash=$1

if [ -z $2 ]
then
    echo "Fork main branch name is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
fork_main_branch=$2

if [ -z $3 ]
then
    echo "Fork name is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
fork_name=$3

from_branch="fbsync"
git stash
git checkout $from_branch
git pull
# Add random prefix in the new branch name to keep it unique per run
prefix=$RANDOM
IFS='
'
for line in $(git log --pretty=oneline "$commit_hash"..HEAD)
do
    if [[ $line != *\[fbsync\]* ]]
    then
        echo "Parsing $line"
        hash=$(echo $line | cut -f1 -d' ')
        git checkout $fork_main_branch
        git checkout -B cherrypick_${prefix}_${hash}
        git cherry-pick -x "$hash"
        git push $fork_name cherrypick_${prefix}_${hash}
        git checkout $from_branch
    fi
done
echo "Please review the PRs, add [FBCode->GH] prefix in the title and publish them."
