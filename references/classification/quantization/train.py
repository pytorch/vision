import datetime
import sys
import os
import time
import copy
import torch
import torch.utils.data
from torch import nn
import torchvision
import torch.quantization
from torchvision import transforms
from .. import utils
from ..train import train_one_epoch, evaluate, load_data

try:
    from apex import amp
except ImportError:
    amp = None


def main(args):
    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')

    data_loader, data_loader_test, train_sampler = load_data(traindir,
                                                             valdir,
                                                             args.batch_size,
                                                             args.eval_batch_size,
                                                             args.workers,
                                                             args.cache_dataset,
                                                             args.distributed)

    print("Creating model", args.model)
    model = torchvision.models.quantization.__dict__[args.model](pretrained_float_model=True)
    if args.test_only is False:
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig(args.backend)
        torch.quantization.prepare_qat(model, inplace=True)
        model.to(device)
        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_step_size,
                                                   gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        model.to(device)
        model.eval()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig(args.backend)
        torch.quantization.prepare(model, inplace=True)
        # Calibrate first
        evaluate(model, criterion, data_loader, device=device,
                 neval_batches=args.num_calibration_batches)
        torch.quantization.convert(model, inplace=True)
        if args.output_dir:
            if utils.is_main_process():
                torch.save(model.state_dict(), os.path.join(args.output_dir,
                           'quantized_post_train_model.pth'))
        print('Saving quantized model')
        evaluate(model, criterion, data_loader_test, device=device)
        return

    model.train()
    model.apply(torch.quantization.enable_observer)
    model.apply(torch.quantization.enable_fake_quant)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        print('Starting training for epoch', epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                        args.print_freq, apex=False)
        lr_scheduler.step()
        with torch.no_grad():
            if epoch >= args.num_observer_update_epochs:
                print('Disabling observer for subseq epochs, epoch = ', epoch)
                model.apply(torch.quantization.disable_observer)
            if epoch >= args.num_batch_norm_update_epochs:
                print('Freezing BN for subseq epochs, epoch = ', epoch)
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print('Evaluate QAT model')

            model.eval()
            evaluate(model, criterion, data_loader_test, device=device)
            quantized_eval_model = copy.deepcopy(model)
            quantized_eval_model.eval()
            quantized_eval_model.to(torch.device('cpu'))
            torch.quantization.convert(quantized_eval_model, inplace=True)

            print('Evaluate Quantized model')
            evaluate(quantized_eval_model, criterion, data_loader_test,
                     device=torch.device('cpu'))

        model.train()

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'eval_model': quantized_eval_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        print('Saving models after epoch ', epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path',
                        default='/datasets01/imagenet_full_size/061417/',
                        help='dataset')
    parser.add_argument('--model',
                        default='mobilenet_v2',
                        help='model')
    parser.add_argument('--backend',
                        default='fbgemm',
                        help='fbgemm or qnnpack')
    parser.add_argument('--device',
                        default='cuda',
                        help='device')

    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='batch size for calibration/training')
    parser.add_argument('--eval-batch-size', default=128, type=int,
                        help='batch size for evaluation')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_observer_update_epochs',
                        default=4, type=int, metavar='N',
                        help='number of total epochs to update observers')
    parser.add_argument('--num_batch_norm_update_epochs', default=3,
                        type=int, metavar='N',
                        help='number of total epochs to update batch norm stats')
    parser.add_argument('--num_calibration_batches',
                        default=32, type=int, metavar='N',
                        help='number of batches of training set for \
                              observer calibration ')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr',
                        default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. \
             It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url',
                        default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
