Utility scripts
===============

* `fbcode_to_main_sync.sh`

This shell script is used to synchronise internal changes with the main repository.

To run this script:

.. code:: bash

    chmod +x fbcode_to_main_sync.sh
    ./fbcode_to_main_sync.sh <commit_hash> <fork_name> <fork_main_branch>

where

``commit_hash`` represents the commit hash in fbsync branch from where we should start the sync.

``fork_name`` is the name of the remote corresponding to your fork, you can check it by doing `"git remote -v"`.

``fork_main_branch`` (optional) is the name of the main branch on your fork(default="main").

This script will create PRs corresponding to the commits in fbsync. Please review these, add the [FBcode->GH] prefix on the title and publish them. Most importantly, add the [FBcode->GH] prefix at the beginning of the merge message as well.
