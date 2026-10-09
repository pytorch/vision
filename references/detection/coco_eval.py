import copy
import io
from contextlib import redirect_stdout

import numpy as np
import torch
import utils


def _load_coco_backend(backend):
    if backend == "pycocotools":
        import pycocotools.mask as mask_util
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    elif backend == "ultrafast":
        try:
            import ultrafast_pycocotools.mask as mask_util
            from ultrafast_pycocotools import COCO, COCOeval
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'The ultrafast backend requires pip install "ultrafast-pycocotools>=0.1.11,<0.2".'
            ) from exc
    else:
        raise ValueError(f"Unknown COCO backend {backend!r}; choose 'pycocotools' or 'ultrafast'.")
    return COCO, COCOeval, mask_util


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types, backend="pycocotools"):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self._coco, cocoeval, mask_util = _load_coco_backend(backend)
        self._encode_mask = mask_util.encode
        if not isinstance(coco_gt, self._coco):
            # Dataset loaders can keep their own COCO implementation.
            converted = self._coco()
            converted.dataset = coco_gt.dataset
            with redirect_stdout(io.StringIO()):
                converted.createIndex()
            coco_gt = converted
        self.coco_gt = coco_gt
        self.backend = backend

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            kwargs = {"store_eval_imgs": True} if backend == "ultrafast" else {}
            self.coco_eval[iou_type] = cocoeval(coco_gt, iouType=iou_type, **kwargs)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self._predictions = {k: {} for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        if not img_ids:
            return
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            if self.backend == "ultrafast":
                # This backend evaluates and accumulates together. Gather predictions
                # before evaluation instead of assigning externally merged evalImgs.
                per_image = {image_id: [] for image_id in img_ids}
                for result in results:
                    per_image[result["image_id"]].append(result)
                for image_id in img_ids:
                    self._predictions[iou_type].setdefault(image_id, per_image[image_id])
                continue
            with redirect_stdout(io.StringIO()):
                coco_dt = self.coco_gt.loadRes(results) if results else self._coco()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            if self.backend == "ultrafast":
                merged = {}
                for per_rank in utils.all_gather(self._predictions[iou_type]):
                    for image_id, results in per_rank.items():
                        merged.setdefault(image_id, results)
                # Match merge(): sorted IDs, first occurrence across ranks/batches.
                image_ids = sorted(merged)
                results = [result for image_id in image_ids for result in merged[image_id]]
                evaluator = self.coco_eval[iou_type]
                with redirect_stdout(io.StringIO()):
                    evaluator.cocoDt = self.coco_gt.loadRes(results) if results else self._coco()
                evaluator.params.imgIds = image_ids
                _, self.eval_imgs[iou_type] = evaluate(evaluator)
                self._predictions[iou_type].clear()
                continue
            params = self.coco_eval[iou_type].params
            self.eval_imgs[iou_type] = (
                np.concatenate(self.eval_imgs[iou_type], 2)
                if self.eval_imgs[iou_type]
                else np.empty((len(params.catIds) if params.useCats else 1, len(params.areaRng), 0), dtype=object)
            )
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                self._encode_mask(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    categories = len(imgs.params.catIds) if imgs.params.useCats else 1
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(
        categories, len(imgs.params.areaRng), len(imgs.params.imgIds)
    )
