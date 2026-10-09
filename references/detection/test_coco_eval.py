import builtins
import copy
import pickle

import coco_eval
import numpy as np
import pytest
import torch

pytest.importorskip("pycocotools")
pytest.importorskip("ultrafast_pycocotools")


def _inputs():
    from pycocotools.coco import COCO

    keypoints = [[3 + i % 4, 4 + i // 4, 2] for i in range(17)]
    ann = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [2, 2, 12, 12],
        "area": 144,
        "segmentation": [[2, 2, 14, 2, 14, 14, 2, 14]],
        "keypoints": np.asarray(keypoints).flatten().tolist(),
        "num_keypoints": 17,
    }
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": i, "width": 32, "height": 32} for i in (1, 2, 3)],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [ann, dict(ann, id=2, image_id=2, iscrowd=1)],
    }
    gt.createIndex()
    masks = torch.zeros((3, 1, 32, 32))
    masks[0, :, 18:30, 18:30] = 1
    masks[1:, :, 2:14, 2:14] = 1
    pose = torch.tensor(keypoints, dtype=torch.float32)
    shifted_pose = pose.clone()
    shifted_pose[:, :2] += 16
    prediction = {
        "boxes": torch.tensor([[18.0, 18.0, 30.0, 30.0], [2.0, 2.0, 14.0, 14.0], [2.0, 2.0, 14.0, 14.0]]),
        "scores": torch.tensor([0.9, 0.8, 0.8]),
        "labels": torch.ones(3, dtype=torch.int64),
        "masks": masks,
        "keypoints": torch.stack([shifted_pose, pose, pose]),
    }
    return gt, {1: prediction, 2: {}, 3: {}}


def _assert_evaluators_equal(candidate, reference):
    assert candidate.img_ids == reference.img_ids
    for iou_type in reference.iou_types:
        actual, expected = candidate.coco_eval[iou_type], reference.coco_eval[iou_type]
        assert actual.params.imgIds == expected.params.imgIds
        for key in ("precision", "recall", "scores"):
            np.testing.assert_array_equal(actual.eval[key], expected.eval[key])
        np.testing.assert_array_equal(actual.stats, expected.stats)
        assert len(actual.evalImgs) == len(expected.evalImgs)
        for a, e in zip(actual.evalImgs, expected.evalImgs):
            if e is None:
                assert a is None
            else:
                assert a.keys() == e.keys()
                for key in e:
                    np.testing.assert_array_equal(a[key], e[key])


def _finish(evaluator):
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()


@pytest.mark.parametrize("iou_types", [["bbox"], ["segm"], ["keypoints"], ["bbox", "segm", "keypoints"]])
def test_full_array_parity_and_lifecycle(iou_types):
    gt, predictions = _inputs()
    original = copy.deepcopy(gt.dataset)
    evaluators = []
    for backend in ("pycocotools", "ultrafast"):
        evaluator = coco_eval.CocoEvaluator(gt, iou_types, backend=backend)
        evaluator.update({1: predictions[1]})
        evaluator.update({2: {}, 3: {}})
        evaluator.update({1: {key: value[:1] for key, value in predictions[1].items()}})
        evaluator = pickle.loads(pickle.dumps(evaluator))
        _finish(evaluator)
        evaluators.append(evaluator)
    _assert_evaluators_equal(evaluators[1], evaluators[0])
    assert evaluators[1].coco_eval[iou_types[0]].params.imgIds == [1, 2, 3]
    assert gt.dataset == original
    for evaluator in evaluators:
        evaluator.accumulate()
        evaluator.summarize()
    _assert_evaluators_equal(evaluators[1], evaluators[0])


def _distributed_worker(rank, init_file, empty_rank):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        gt, predictions = _inputs()
        iou_types = ["bbox", "segm", "keypoints"]
        evaluators = []
        for backend in ("pycocotools", "ultrafast"):
            evaluator = coco_eval.CocoEvaluator(gt, iou_types, backend=backend)
            if empty_rank:
                evaluator.update(predictions if rank == 0 else {})
            elif rank == 0:
                evaluator.update({1: predictions[1]})
            else:
                padded = predictions.copy()
                padded[1] = {key: value[:1] for key, value in predictions[1].items()}
                evaluator.update(padded)
            _finish(evaluator)
            evaluators.append(evaluator)
        _assert_evaluators_equal(evaluators[1], evaluators[0])
        reference = coco_eval.CocoEvaluator(gt, iou_types)
        reference.update(predictions)
        for iou_type in iou_types:
            images = np.concatenate(reference.eval_imgs[iou_type], 2)
            target = reference.coco_eval[iou_type]
            target.evalImgs = list(images.flatten())
            target.params.imgIds = [1, 2, 3]
            target._paramsEval = copy.deepcopy(target.params)
        reference.accumulate()
        reference.summarize()
        for iou_type in iou_types:
            for key in ("precision", "recall", "scores"):
                np.testing.assert_array_equal(
                    evaluators[1].coco_eval[iou_type].eval[key], reference.coco_eval[iou_type].eval[key]
                )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("empty_rank", [False, True])
def test_distributed_merge(tmp_path, empty_rank):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "gloo"), empty_rank), nprocs=2, join=True)


def test_backend_errors_and_isolation(monkeypatch):
    gt, _ = _inputs()
    with pytest.raises(ValueError, match="Unknown COCO backend"):
        coco_eval.CocoEvaluator(gt, ["bbox"], backend="invalid")
    original_import = builtins.__import__

    def no_ultrafast(name, *args, **kwargs):
        if name.startswith("ultrafast_pycocotools"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultrafast)
    with pytest.raises(ModuleNotFoundError, match="ultrafast-pycocotools>=0.1.11"):
        coco_eval.CocoEvaluator(gt, ["bbox"], backend="ultrafast")
    assert type(coco_eval.CocoEvaluator(gt, ["bbox"]).coco_gt).__module__ == "pycocotools.coco"


def test_training_argument():
    import train

    args = train.get_args_parser().parse_args(["--coco-backend", "ultrafast", "--test-only"])
    assert args.coco_backend == "ultrafast"
    assert train.get_args_parser().parse_args([]).coco_backend == "pycocotools"


def test_engine_backend_forwarding(monkeypatch):
    import engine
    import torchvision

    gt, predictions = _inputs()

    class Dataset(torchvision.datasets.CocoDetection):
        def __init__(self):
            self.coco = gt

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.tensor([index + 1]), {"image_id": index + 1}

    class Model(torch.nn.Module):
        def forward(self, images):
            return [predictions[int(image[0])] for image in images]

    monkeypatch.setattr(engine, "_get_iou_types", lambda model: ["bbox", "segm", "keypoints"])
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=2, collate_fn=coco_eval.utils.collate_fn)
    reference = engine.evaluate(Model(), loader, torch.device("cpu"))
    candidate = engine.evaluate(Model(), loader, torch.device("cpu"), coco_backend="ultrafast")
    _assert_evaluators_equal(candidate, reference)
