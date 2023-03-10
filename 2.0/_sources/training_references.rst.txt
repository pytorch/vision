Training references
===================

On top of the many models, datasets, and image transforms, Torchvision also
provides training reference scripts. These are the scripts that we use to train
the :ref:`models <models>` which are then available with pre-trained weights.

These scripts are not part of the core package and are instead available `on
GitHub <https://github.com/pytorch/vision/tree/main/references>`_. We currently
provide references for
`classification <https://github.com/pytorch/vision/tree/main/references/classification>`_,
`detection <https://github.com/pytorch/vision/tree/main/references/detection>`_,
`segmentation <https://github.com/pytorch/vision/tree/main/references/segmentation>`_,
`similarity learning <https://github.com/pytorch/vision/tree/main/references/similarity>`_,
and `video classification <https://github.com/pytorch/vision/tree/main/references/video_classification>`_.

While these scripts are largely stable, they do not offer backward compatibility
guarantees.

In general, these scripts rely on the latest (not yet released) pytorch version
or the latest torchvision version. This means that to use them, **you might need
to install the latest pytorch and torchvision versions**, with e.g.::

    conda install pytorch torchvision -c pytorch-nightly

If you need to rely on an older stable version of pytorch or torchvision, e.g.
torchvision 0.10, then it's safer to use the scripts from that corresponding
release on GitHub, namely
https://github.com/pytorch/vision/tree/v0.10.0/references.
