#!/bin/bash

if [ -z $1 ]
then
    echo "From repo is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <from_repo> <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
from_repo=$1

if [ -z $2 ]
then
    echo "Commit hash is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <from_repo> <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
commit_hash=$2

if [ -z $3 ]
then
    echo "Fork main branch name is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <from_repo> <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
fork_main_branch=$3

if [ -z $4 ]
then
    echo "Fork name is required to be passed when running this script."
    echo "./fbcode_to_main_sync.sh <from_repo> <commit_hash> <fork_main_branch> <fork_name>"
    exit 1
fi
fork_name=$4

git stash
git checkout $from_repo
git fetch
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
        git checkout $from_repo
    fi
done
echo "Please review the PRs, add [FBCode->GH] tag on the title and publish them."
