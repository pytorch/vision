import pytest


def test_warns_if_imported_from_datasets():
    import torchvision

    value = torchvision._WARN_ABOUT_BETA_TRANSFORMS

    setattr(torchvision, "_WARN_ABOUT_BETA_TRANSFORMS", True)

    with pytest.warns(UserWarning, match=torchvision._BETA_TRANSFORMS_WARNING):
        from torchvision.datasets import wrap_dataset_for_transforms_v2

    setattr(torchvision, "_WARN_ABOUT_BETA_TRANSFORMS", value)


def test_no_warns_if_imported_from_datasets():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from torchvision.datasets import wrap_dataset_for_transforms_v2
