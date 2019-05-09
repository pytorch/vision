import utils
import copy
import torch
import time

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_width, ratio_height = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        image_height, image_width = prediction["original_image_size"]

        boxes = prediction["boxes"]
        c_height, c_width = prediction["image_size"]
        boxes = resize_boxes(boxes, (c_width, c_height), (image_width, image_height))
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


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    tic = time.time()
    #print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if not p.useSegm is None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params=p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                    for imgId in p.imgIds
                    for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
             for catId in catIds
             for areaRng in p.areaRng
             for imgId in p.imgIds
         ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

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

import numpy as np
def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)



import json
import tempfile
class CocoEvaluator(object):

    def __init__(self, coco_gt, iou_type):

        # from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self.coco_gt = coco_gt
        self.coco_eval = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []#sorted(coco_gt.getImgIds())
        self.eval_imgs = []


    def append(self, predictions):
        if -1 in predictions:
            del predictions[-1]
        img_ids = list(predictions.keys())
        results = prepare_for_coco_detection(predictions)

        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            with open(file_path, "w") as f:
                json.dump(results, f)

            coco_dt = self.coco_gt.loadRes(str(file_path))
            self.coco_eval.cocoDt = coco_dt
            self.coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(self.coco_eval)
            self.img_ids.extend(img_ids)
            # self.eval_imgs.extend(eval_imgs)
            self.eval_imgs.append(eval_imgs)

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        # self.eval_imgs = list(self.eval_imgs.flatten())
        create_common_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        self.coco_eval.summarize()

    # def __repr__(self):
    #     return repr(self.coco_eval)

