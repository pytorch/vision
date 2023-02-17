import pytest


def test_warns_if_imported_from_datasets(mocker):
    mocker.patch("torchvision._WARN_ABOUT_BETA_TRANSFORMS", return_value=True)

    import torchvision

    with pytest.warns(UserWarning, match=torchvision._BETA_TRANSFORMS_WARNING):
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        assert callable(wrap_dataset_for_transforms_v2)


@pytest.mark.filterwarnings("error")
def test_no_warns_if_imported_from_datasets():
    from torchvision.datasets import wrap_dataset_for_transforms_v2

    assert callable(wrap_dataset_for_transforms_v2)
