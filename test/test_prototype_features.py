import pytest
import torch
from prototype_common_utils import make_image, make_label, make_segmentation_mask
from torch.utils.data import DataLoader
from torchvision.prototype import features


def test_isinstance():
    assert isinstance(
        features.Label([0, 1, 0], categories=["foo", "bar"]),
        torch.Tensor,
    )


class TestTorchFunction:
    def test_wrapping_no_copy(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        assert label.data_ptr() == tensor.data_ptr()

    def test_to_wrapping(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        label_to = label.to(torch.int32)

        assert type(label_to) is features.Label
        assert label_to.dtype is torch.int32
        assert label_to.categories is label.categories

    def test_to_feature_reference(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"]).to(torch.int32)

        tensor_to = tensor.to(label)

        assert type(tensor_to) is torch.Tensor
        assert tensor_to.dtype is torch.int32

    def test_clone_wrapping(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        label_clone = label.clone()

        assert type(label_clone) is features.Label
        assert label_clone.data_ptr() != label.data_ptr()
        assert label_clone.categories is label.categories

    def test_requires_grad__wrapping(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.float32)
        label = features.Label(tensor, categories=["foo", "bar"])

        assert not label.requires_grad

        label_requires_grad = label.requires_grad_(True)

        assert type(label_requires_grad) is features.Label
        assert label.requires_grad
        assert label_requires_grad.requires_grad

    def test_other_op_no_wrapping(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        # any operation besides .to() and .clone() will do here
        output = label * 2

        assert type(output) is torch.Tensor

    @pytest.mark.parametrize(
        "op",
        [
            lambda t: t.numpy(),
            lambda t: t.tolist(),
            lambda t: t.max(dim=-1),
        ],
    )
    def test_no_tensor_output_op_no_wrapping(self, op):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        output = op(label)

        assert type(output) is not features.Label

    def test_inplace_op_no_wrapping(self):
        tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        label = features.Label(tensor, categories=["foo", "bar"])

        output = label.add_(0)

        assert type(output) is torch.Tensor
        assert type(label) is features.Label


def test_new_like():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    # any operation besides .to() and .clone() will do here
    output = label * 2

    label_new = features.Label.new_like(label, output)

    assert type(label_new) is features.Label
    assert label_new.data_ptr() == output.data_ptr()
    assert label_new.categories is label.categories


class TestVisionCollate:
    def check_collation(self, dataset, expected_batch, *, collate_fn=features.vision_collate):
        data_loader = DataLoader(dataset, num_workers=0, batch_size=len(dataset), collate_fn=collate_fn)

        actual_batch = list(data_loader)[0]

        torch.testing.assert_close(actual_batch, expected_batch)

        return actual_batch

    @pytest.mark.parametrize("with_labels", [True, False])
    def test_classification(self, with_labels):
        image_size = (16, 17)
        categories = ["foo", "bar", "baz"]

        dataset = []
        for _ in range(4):
            image = make_image(size=image_size)
            label = make_label(categories=categories) if with_labels else None

            dataset.append((image, label))

        expected_images, expected_labels = zip(*dataset)
        expected_batch = [
            features.Image.new_like(expected_images[0], torch.stack(expected_images)),
            features.Label.new_like(expected_labels[0], torch.stack(expected_labels))
            if with_labels
            else list(expected_labels),
        ]

        actual_batch = self.check_collation(dataset, expected_batch)

        if with_labels:
            assert actual_batch[1].categories == categories

    def test_segmentation(self):
        image_size = (16, 17)

        dataset = []
        for _ in range(4):
            image = make_image(size=image_size)
            mask = make_segmentation_mask(size=image_size, num_categories=10)

            dataset.append((image, mask))

        expected_batch = [
            type(expected_features[0]).new_like(expected_features[0], torch.stack(expected_features))
            for expected_features in zip(*dataset)
        ]

        self.check_collation(dataset, expected_batch)
