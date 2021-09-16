import pytest
from torchvision.prototype.datasets.utils import DatasetInfo


@pytest.fixture
def make_minimal_dataset_info():
    def make(name="name", categories=None, **kwargs):
        return DatasetInfo(name, categories=categories or [], **kwargs)

    return make
