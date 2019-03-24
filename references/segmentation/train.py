import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

import datasets
import models
import utils


def get_dataset(name, image_set, transform):
    paths = {
        "voc": ('/datasets01/VOC/060817/', datasets.VOC),
        "voc_aug": ('/datasets01/SBDD/072318/', datasets.SBDD),
        "coco": ('/datasets01/COCO/060817/', datasets.COCO)
    }
    p, ds_fn = paths[name]

    return ds_fn(p, image_set, transforms=transform)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for image, target in metric_logger.log_every(data_loader, 100, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        output = output['out']

        confmat.update(target.flatten(), output.argmax(1).flatten())

    confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item())


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    torch_save = torch.save
    def save(*args, **kwargs):
        if is_master:
            torch_save(*args, **kwargs)
    torch.save = save

def main(args):
    args.gpu = args.local_rank

    if args.distributed:
        args.rank = int(os.environ["RANK"])
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        dist_url = 'env://'
        print('| distributed init (rank {}): {}'.format(
            args.rank, dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=dist_url)
        suppress_output(args.rank == 0)

    device = torch.device(args.device)

    dataset = get_dataset(args.dataset, "train", datasets.Transform(0.5))
    dataset_test = get_dataset(args.dataset, "val", datasets.Transform())

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers, collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)

    model = models.get_model(args.model, args.backbone, num_classes=dataset.num_classes, aux=args.aux_loss)
    model.to(device)
    model = torch.nn.utils.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.head['out'].parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params_to_optimize.append(
            {"params": [p for p in model_without_ddp.head['aux'].parameters() if p.requires_grad], "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        with torch.no_grad():
            confmat = evaluate(model, data_loader_test, device=device, num_classes=dataset.num_classes)
        print(confmat)
        torch.save({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--dataset', default='voc', help='dataset')
    parser.add_argument('--model', default='fcn', help='model')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--local_rank', default=0, type=int, help='print frequency')

    args = parser.parse_args()
    print(args)

    if args.output_dir:
        utils.mkdir(args.output_dir)


    import os
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    main(args)
