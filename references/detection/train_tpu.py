r"""
Basic training script for Training MaskRCNN on TPUs.
"""
import argparse
import contextlib
import logging
import os
import random
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch_xla
import torch_xla.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
#import torch_xla.data_parallel as dp
import torchvision
# From references/detection/train.py
from coco_utils import get_coco, get_coco_kp
from group_by_aspect_ratio import GroupedBatchSampler
from engine import train_one_epoch, evaluate
import utils
import transforms as T
logger = logging.getLogger("detection.tpu_trainer")
logger.info("Training MaskRCNN")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@contextlib.contextmanager
def _bench(name):
    """Logs the time taken to execute a code block."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f'{name}: {elapsed:.3f}s')


def get_dataset(name, image_set, transform):
    paths = {
        "coco": ('/home/miladmo/datasets01/COCO', get_coco, 91),
        "coco_kp": ('/coco/', get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='test',
        choices=('eval', 'test', 'train'),
        help='eval computes the AP scores, test runs inference on a few sample images, '
        'train trains the model')
    parser.add_argument(
        '--num-eval', type=int, default=50,
        help='Number of validation set images to use in eval mode')
    parser.add_argument(
        '--num_cores', type=int, default=1,
        help='Number of TPU cores to train on (8 for single v2/v3 Cloud TPU)')
    parser.add_argument(
        "--use_cpu",
        help="Whether to use cpu instead of tpu",
        action="store_true")
    args = parser.parse_args()
    return args
def random_colour_masks(mask):
    colours = [
	[0, 255, 0],
	[0, 0, 255],
	[255, 0, 0],
	[0, 255, 255],
	[255, 255, 0],
	[255, 0, 255],
	[80, 70, 180],
	[250, 80, 190],
	[245, 145, 50],
	[70, 150, 250],
	[50, 190, 190]
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(3,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def overlay_output(img, masks, boxes, pred_cls, alpha=0.5):
    for i in range(len(masks)):
        draw = ImageDraw.Draw(img)
        draw.rectangle(boxes[i])
        draw.text(boxes[i][0], pred_cls[i], 'black')
        rgb_mask = random_colour_masks(masks[i])
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*rgb_mask)).convert('L')
        img = Image.composite(pil_mask, img, pil_mask)
    return img

def process_prediction(output, threshold=0.05):
    output = output[0]
    pred_score = list(output['scores'].cpu().clone().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = ((output['masks'].cpu().clone())>0.5).squeeze().detach().numpy()
    classes = output['labels'].cpu().clone().detach().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in classes]
    boxes = output['boxes'].cpu().clone().detach().numpy()
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def do_prediction(image_path, model, use_cpu=False):
    image = Image.open(image_path)
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    with _bench('inference'):
        print("milad: do model call on image list")
        output = model([image_tensor])
        print("milad: done model call on image list")

    xm.master_print('# of boxes: {}'.format(len(output[0]['boxes'])))

    with _bench('cpu postprocessing'):
        masks, pred_boxes, pred_class = process_prediction(output)
        image = overlay_output(image, masks, pred_boxes, pred_class)
        image.save(os.path.join('/home/miladmo/datasets01/COCO/out', os.path.basename(image_path)))

    xm.master_print(torch_xla._XLAC._xla_metrics_report())
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    #model.eval()

    #image = Image.open(image_path)
    #image = torchvision.transforms.functional.resize(image, (800, 600))
    #image_tensor = torchvision.transforms.functional.to_tensor(image)

    #if not use_cpu:
    #    # Send to single TPU device
    #    device = xm.xla_device()
    #    xm.mark_step()
    #    #torch_xla._XLAC._xla_set_default_device(str(device))
    #    model = model.to(device)
    #    image_tensor = image_tensor.to(device)
    #output = model([image_tensor])
    #xm.mark_step()

    #print("output: {}".format(output))
    ##print(torch_xla._XLAC._xla_metrics_report())
    #masks, pred_boxes, pred_class = process_prediction(output)
    #image = overlay_output(image, masks, pred_boxes, pred_class)
    #image.save(os.path.join('/home/miladmo/datasets01/COCO/out', os.path.basename(image_path)))

def train_one_epoch_tpu(
    model, criterion, optimizer, data_loader, device, epoch, print_freq
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


def do_train(num_cores):
    # Default (h)params
    num_workers = 0
    num_epochs = 1
    print_freq = 1
    batch_size = 1
    lr = 0.02
    lr_steps = [8, 11]
    lr_gamma = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    logger.info("Loading data")
    dataset, num_classes = get_dataset('coco', "train", get_transform(train=True))
    dataset_test, _ = get_dataset('coco', "val", get_transform(train=False))

    logger.info("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    logger.info("Creating model and sending to XLA device")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    model.train()
    device = xm.xla_device() #'xla:0' #
    torch_xla._XLAC._xla_set_default_device(str(device))
    model = model.to(device)

    logger.info("Creating optimizer and lr scheduler")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_steps, gamma=lr_gamma)


    for epoch in range(1, num_epochs+1):
        logger.info("Training epoch: {epoch}".format(epoch=epoch))
        train_one_epoch_tpu(
            model, optimizer, data_loader, device, epoch, print_freq)
        # Step lr every epoch
        lr_scheduler.step()


def main():
    FLAGS = parse_flags()
    logger.info("FLAGS: {}".format(FLAGS))

    def mark_step():
        if not FLAGS.use_cpu:
            with _bench('mark_step'):
                xm.mark_step()

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, is_xla=True)
    print("milad: do model eval")
    model.eval()
    print("milad: done model eval")

    device = xm.xla_device() #'xla:0' #
    if not FLAGS.use_cpu:
        model = model.to(device)

    if FLAGS.mode == 'eval':
        dataset_val, _ = get_dataset('coco', 'val', T.ToTensor())
        dataset_val = torch.utils.data.Subset(dataset_val, range(FLAGS.num_eval))

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

        evaluate(model, data_loader_val, device=device, move_tensors_to_device=False)
    elif FLAGS.mode == 'test':
        mark_step()
        print("milad: do_prediction a")
        a = do_prediction('/home/miladmo/datasets01/COCO/train2017/000000000308.jpg', model, use_cpu=FLAGS.use_cpu)
        print("milad: done do_prediction a")
        mark_step()
        print("milad: done mask_step a")
        b = do_prediction('/home/miladmo/datasets01/COCO/train2017/000000000394.jpg', model, use_cpu=FLAGS.use_cpu)
        mark_step()
        c = do_prediction('/home/miladmo/datasets01/COCO/train2017/000000000326.jpg', model, use_cpu=FLAGS.use_cpu)
        mark_step()
        d = do_prediction('/home/miladmo/datasets01/COCO/train2017/000000000368.jpg', model, use_cpu=FLAGS.use_cpu)
    elif FLAGS.mode == 'train':
        raise NotImplementedError

    #do_prediction('/home/miladmo/datasets01/COCO/train2017/000000566234.jpg', use_cpu=FLAGS.use_cpu)
    #do_train(FLAGS.num_cores)
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
