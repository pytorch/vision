Training references
===================

On top of the many models, datasets, and image transforms, Torchvision also
provides training reference scripts. These are the scripts that we use to train
the :ref:`models <models>` which are then available with pre-trained weights.

These scripts are not part of the core package and are instead available `on
GitHub <https://github.com/pytorch/vision/tree/main/references>`_.

In general, these scripts rely on the latest (not yet released) pytorch version
or the latest torchvision version. This means that to use them, **you might need
to install the latest pytorch and torchvision versions**, with e.g.::

    conda install pytorch torchvision -c pytorch-nightly

If you need to rely on an older stable version of pytorch or torchvision, e.g.
torchvision 0.10, then it's safer to use the scripts from that corresponding
release on GitHub, namely
https://github.com/pytorch/vision/tree/v0.10.0/references.

Please also note that while these scripts are largely stable, they do not have
backward compatibility guarantees.
