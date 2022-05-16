from typing import List, Callable

import pytest
import torch
import torch.fx
from torch import Tensor
from torchvision import ops


class IouTestBase:
    @staticmethod
    def _run_test(target_fn: Callable, test_input: List, dtypes: List[torch.dtype], tolerance: float, expected: List):
        def assert_close(box: Tensor, expected: Tensor, tolerance):
            out = target_fn(box, box)
            torch.testing.assert_close(out, expected, rtol=0.0, check_dtype=False, atol=tolerance)

        for dtype in dtypes:
            actual_box = torch.tensor(test_input, dtype=dtype)
            expected_box = torch.tensor(expected)
            assert_close(actual_box, expected_box, tolerance)

    @staticmethod
    def _run_jit_test(target_fn: Callable, test_input: List):
        box_tensor = torch.tensor(test_input, dtype=torch.float)
        expected = target_fn(box_tensor, box_tensor)
        scripted_fn = torch.jit.script(target_fn)
        scripted_out = scripted_fn(box_tensor, box_tensor)
        torch.testing.assert_close(scripted_out, expected, rtol=0.0, atol=1e-3)


def _generate_int_input():
    return [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]


def _generate_float_input():
    return [
        [285.3538, 185.5758, 1193.5110, 851.4551],
        [285.1472, 188.7374, 1192.4984, 851.0669],
        [279.2440, 197.9812, 1189.4746, 849.2019],
    ]


class TestBoxIou(IouTestBase):
    def _generate_int_expected():
        return [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _generate_float_input():
        return [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]

    def _generate_float_expected():
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32, torch.float64], 1e-3, _generate_float_expected()),
        ],
    )
    def test_iou(self, test_input, dtypes, tolerance, expected):
        self._run_test(ops.box_iou, test_input, dtypes, tolerance, expected)

    def test_iou_jit(self):
        self._run_jit_test(ops.box_iou, [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]])


class TestGenBoxIou(IouTestBase):
    def _generate_int_expected():
        return [[1.0, 0.25, -0.7778], [0.25, 1.0, -0.8611], [-0.7778, -0.8611, 1.0]]

    def _generate_float_expected():
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32, torch.float64], 1e-3, _generate_float_expected()),
        ],
    )
    def test_iou(self, test_input, dtypes, tolerance, expected):
        self._run_test(ops.generalized_box_iou, test_input, dtypes, tolerance, expected)

    def test_iou_jit(self):
        self._run_jit_test(ops.generalized_box_iou, [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]])


class TestDistanceBoxIoU(IouTestBase):
    def _generate_int_expected():
        return [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _generate_float_expected():
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32, torch.float64], 1e-3, _generate_float_expected()),
        ],
    )
    def test_iou(self, test_input, dtypes, tolerance, expected):
        self._run_test(ops.distance_box_iou, test_input, dtypes, tolerance, expected)

    def test_iou_jit(self):
        self._run_jit_test(ops.distance_box_iou, [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]])


class TestCompleteBoxIou(IouTestBase):
    def _generate_int_expected():
        return [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _generate_float_expected():
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32, torch.float64], 1e-3, _generate_float_expected()),
        ],
    )
    def test_iou(self, test_input, dtypes, tolerance, expected):
        self._run_test(ops.complete_box_iou, test_input, dtypes, tolerance, expected)

    def test_iou_jit(self):
        self._run_jit_test(ops.complete_box_iou, [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]])


if __name__ == "__main__":
    pytest.main([__file__])
