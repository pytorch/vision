import datetime
import os
import sys
import math
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision import transforms

from coco_utils import get_coco

import utils
import transforms as T

from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval2 import do_coco_evaluation as _evaluate


def get_dataset(name, image_set, transform):
    paths = {
        # "voc": ('/datasets01/VOC/060817/', torchvision.datasets.VOCSegmentation, 21),
        # "voc_aug": ('/datasets01/SBDD/072318/', sbd, 21),
        "coco": ('/datasets01/COCO/022719/', get_coco, 81)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    min_size = 800
    max_size = 1333
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    if True:
        transforms.append(T.BGR255())
        transforms.append(T.Normalize(mean=[102.9801, 115.9465, 122.7717],
                                      std=[1., 1., 1.]))
    else:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if epoch == 0:
        warmup_factor = 1. / 3
        warmup_iters = 500
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if epoch == 0:
            lr_scheduler.step()

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def evaluate(model, criterion, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    results_dict = utils.Dict()
    CAT_LIST = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
        79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90], device=device)
    with torch.no_grad():
        for image, targets in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            for o, t in zip(outputs, targets):
                o["original_image_size"] = t["original_image_size"]
                o["labels"] = CAT_LIST[o["labels"]]

            results_dict.update(
                {target["image_id"][0].item(): output for target, output in zip(targets, outputs)})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results_dict.synchronize_between_processes()
    if -1 in results_dict:
        del results_dict[-1]

    extra_args = dict(
        box_only=False,
        iou_types=("bbox",),
        expected_results=None,
        expected_results_sigma_tol=None,
    )
    output_folder = None

    dataset = data_loader.dataset
    while True:
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        dataset = dataset.dataset

    return _evaluate(dataset=dataset,
                    predictions=results_dict,
                    output_folder=output_folder,
                    **extra_args)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    args.dataset = "coco"
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, drop_last=True,
        collate_fn=utils.BatchCollator(32))

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,#args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.BatchCollator(32))

    print("Creating model")
    # model = torchvision.models.__dict__[args.model]()
    model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device)
    # if args.distributed:
    #     model = torch.nn.utils.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict
        # state_dict = torch.load('/checkpoint/fmassa/jobs/detectron_logs/detectron_12296927/model_final.pth')
        # state_dict = torch.load('maskrcnn-benchmark/model_final.pth')
        # load_state_dict(model, state_dict['model'])
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr_scheduler.step()
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        if True: # epoch == args.epochs - 1:
            evaluate(model, criterion, data_loader_test, device=device)
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
