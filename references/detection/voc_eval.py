from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import shutil
import pickle
import numpy as np
import pdb
from torchvision.datasets import VOCDetection

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(classname,
             detpath,
             imagesetfile,
             annopath='',
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    recs = {}
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annotations
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
      # sort by confidence
      sorted_ind = np.argsort(-confidence)
      sorted_scores = np.sort(-confidence)
      BB = BB[sorted_ind, :]
      image_ids = [image_ids[x] for x in sorted_ind]

      # go down dets and mark TPs and FPs
      for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                 (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                 (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def _write_voc_results_file(all_boxes, image_index, root, classes):
    if os.path.exists('/tmp/results'):
        shutil.rmtree('/tmp/results')
    os.makedirs('/tmp/results')
    print('Writing results file', end='\r')
    for cls_ind, cls  in enumerate(classes):
        # DistributeSampler happens to clone the inputs to make the task 
        # lenghts even among the nodes:
        # https://github.com/pytorch/pytorch/issues/22584
        # Boxes can be duplicated in the process since multiple
        # evaluation of the same image can happen, multiple boxes in the
        # same location decrease the final mAP, later in the code we discard
        # repeated image_index thanks to the sorting
        new_image_index, all_boxes[cls_ind] = zip(*sorted(zip(image_index,
                                 all_boxes[cls_ind]), key=lambda x: x[0]))
        if cls == '__background__':
            continue

        filename = '/tmp/results/det_test_{:s}.txt'.format(cls)
        with open(filename, 'wt') as f:
            prev_index = ''
            for im_ind, index in enumerate(new_image_index):
                # check for repeated input and discard
                if prev_index == index: continue
                prev_index = index
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                dets = dets[0]
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def _do_python_eval(data_loader):
    imagesetfile = os.path.join(data_loader.dataset.root,
        'VOCdevkit/VOC2007/ImageSets/Main/'+data_loader.dataset.image_set+'.txt')
    annopath = os.path.join(data_loader.dataset.root,
        'VOCdevkit/VOC2007/Annotations/{:s}.xml')

    classes = data_loader.dataset._transforms.transforms[0].CLASSES
    aps = []
    for cls in classes:
        if cls == '__background__':    
            continue    
        filename = '/tmp/results/det_test_{:s}.txt'.format(cls)    
        rec, prec, ap = voc_eval(cls, filename, imagesetfile, annopath,
                            ovthresh=0.5, use_07_metric=True)    
        aps += [ap]
    print('Mean AP = {:.4f}        '.format(np.mean(aps)))
