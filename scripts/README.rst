Utility scripts
===============

* `fbcode_to_main_sync.sh`

This shell script is used to synchronise internal changes with the main repository.

To run this script:

.. code:: bash

    chmod +x fbcode_to_main_sync.sh
    ./fbcode_to_main_sync.sh <from_repo> <commit_hash> <fork_main_branch> <fork_name>

where

``from_repo`` represents the name of the repository from which commits are to be cherry-picked.

``commit_hash`` represents the commit hash in fbsync branch from where we should start the sync.

``fork_main_branch`` is the name of the main branch on your fork.

``fork_name`` is the name of the remote corresponding to your fork, you can check it by doing `"git remote -v"`.

Example:

.. code:: bash

    ./fbcode_to_main_sync.sh fbsync <commit_hash> main origin

This script will create PRs corresponding to the PRs in your ``from_repo`` (fbsync). Please review these, add the [FBCode->GH] prefix on the title and publish them. Most importantly, add the [FBCode->GH] prefix at the beginning of the merge message as well.
