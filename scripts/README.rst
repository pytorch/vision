Utility scripts
===============

* `fbcode_to_main_sync.sh`

This shell script is used to synchronise internal changes with the main repository.

To run this script:

.. code:: bash

    chmod +x fbcode_to_main_sync.sh
    ./fbcode_to_main_sync.sh <commit_hash>

where ``commit_hash`` represents the commit hash of the PR in fbsync branch from where we should start the sync.

This script will create PRs corresponding to the PRs in fbsync. Please review these, add [FBCode->GH] tag on the title and publish them. Remember to add [FBCode->GH] tag at the beginning of the merge message as well.
