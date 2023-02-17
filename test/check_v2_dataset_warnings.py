import warnings

import torchvision


def test_warns_if_imported_from_datasets():
    with warnings.catch_warnings(record=True) as w:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        assert callable(wrap_dataset_for_transforms_v2)

        assert len(w) == 2
        assert "torchvision.transforms.v2" in str(w[-1].message)


def test_no_warns_if_imported_from_datasets():

    torchvision.disable_beta_transforms_warning()

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        from torchvision.datasets import wrap_dataset_for_transforms_v2

        assert callable(wrap_dataset_for_transforms_v2)

        from torchvision.datasets import cifar

        assert hasattr(cifar, "CIFAR10")


if __name__ == "__main__":
    # We can't rely on pytest due to various side-effects, e.g. conftest etc
    test_warns_if_imported_from_datasets()
    test_no_warns_if_imported_from_datasets()
