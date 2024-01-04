'''
    adapted from: https://github.com/facebookresearch/moco/blob/master/main_lincls.py
    
    CUDA_VISIBLE_DEVICES='1' python main_lincls_onecycle.py ipcl1 fc7 --gpu 0 
    CUDA_VISIBLE_DEVICES='1' python main_lincls_onecycle.py ipcl1 fc7 --gpu 0 --evaluate    

'''
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import pandas as pd
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models
from lib import nethook
from lib.one_cycle_scheduler import OneCycleScheduler

from IPython.core.debugger import set_trace

from lib.gpu import is_temp_safe 

model_names = sorted(name for name in models.__dict__
    if name.islower() and name.startswith("ipcl")
    and callable(models.__dict__[name]))

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('arch', metavar='ARCH', choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names))
# parser.add_argument('layer_name', metavar='LAYERNAME', type=str, default=None)
# parser.add_argument('--cut', metavar='CUT', type=int, default=None,
#                     help='number of children to cut off the backbone model, e.g., -1 to drop the fc head of a resnet')
# parser.add_argument('--data', metavar='DIR', help='path to dataset',
#                     default='/home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256')
# parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
#                     help='number of data loading workers (default: 32)')
# parser.add_argument('--epochs', default=10, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=30., type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--div_factor', default=1000000., type=float, help='initial learning rate= lr/div_factor')
# parser.add_argument('--pct_start', default=.30, type=float, help='percent of epochs for rise to lr')
# parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
#                     help='learning rate schedule (when to drop lr by a ratio)')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=0., type=float,
#                     metavar='W', help='weight decay (default: 0.)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=250, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
# 
# parser.add_argument('--checkpoint', default='', type=str,
#                     help='path to pretrained checkpoint')

best_acc1 = 0

class LinearReadout(nn.Module):

    def __init__(self, in_features, num_classes: int = 1000) -> None:
        super(LinearReadout, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('arch', metavar='ARCH', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('layer_name', metavar='LAYERNAME', type=str, default=None)
    parser.add_argument('--cut', metavar='CUT', type=int, default=None,
                        help='number of children to cut off the backbone model, e.g., -1 to drop the fc head of a resnet')
    parser.add_argument('--data', metavar='DIR', help='path to dataset',
                        default='/home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--div_factor', default=1000000., type=float, help='initial learning rate= lr/div_factor')
    parser.add_argument('--pct_start', default=.30, type=float, help='percent of epochs for rise to lr')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=250, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument('--checkpoint', default='', type=str,
                        help='path to pretrained checkpoint')
    
    args = parser.parse_args(raw_args)
        
    if args.resume == '':
        try:
            os.mkdir('./weights')
        except FileExistsError:
            pass
        args.resume = f'./weights/{args.arch}_{args.layer_name}_lincls_onecycle.pth.tar'
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)        
    
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    pprint(args)
    
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> loading feature extractor '{}'".format(args.arch))
    model, _ = models.__dict__[args.arch]()    
    normalize = model.val_transform.transforms[-1]
    
    if args.cut is not None:
        model = nn.Sequential(*list(model.children())[:args.cut])
    else:
        # hook the model layer being used for readout
        if not isinstance (model, nethook.InstrumentedModel):
            print('Wrapping model in InstrumentedModel')
            model = nethook.InstrumentedModel(model) 
        model.retain_layers([args.layer_name])
    print(model)
    
    # freeze all layers
    for name, param in model.named_parameters():
        param.requires_grad = False
        #if name not in ['fc.weight', 'fc.bias']:
        #    param.requires_grad = False
    # init the fc layer
    # model.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.fc.bias.data.zero_()
    
    # determine number output dimensions
    model.eval()
    with torch.no_grad():
        x = model(torch.rand(10,3,224,224))
        if args.cut is None:
            x = model.retained_layer(args.layer_name)
        args.in_features = torch.flatten(x,1).shape[-1]
        
    # create the linear readout classifier
    linear_model = LinearReadout(args.in_features, num_classes=1000)
    print(linear_model)
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        linear_model.to(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            linear_model = torch.nn.DataParallel(linear_model).cuda()
            
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, linear_model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)        
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            linear_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    #normalize = model.model.val_transform.transforms[-1]
    
    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ]
    if isinstance(normalize, transforms.Normalize): train_transforms.append(normalize)
        
    train_dataset = datasets.ImageFolder(traindir, transform=transforms.Compose(train_transforms))
    print(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize,
    ]
    if isinstance(normalize, transforms.Normalize): val_transforms.append(normalize)
        
    val_dataset = datasets.ImageFolder(valdir, transform=transforms.Compose(val_transforms))
    print(val_dataset)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        filename = os.path.join('results', 'main_onecycle', f'{args.arch}_{args.layer_name}_onecycle.csv')
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            print(f"file exists, skipping: {filename}")
            print(df)
        else:
            top1 = validate(val_loader, model, linear_model, criterion, args)
            df = pd.DataFrame(columns=['train_script','model_name','layer_name','epoch','top1'])
            df = df.append({
                "train_script": "main_onecycle.py",
                "model_name": args.arch,
                "layer_name": args.layer_name,
                "epoch": args.start_epoch,
                "top1": top1.item(),
            }, ignore_index=True)
            df.to_csv(filename, index=False)
            print(df)

        return
    
    # setup the one_cycle scheduler
    scheduler = OneCycleScheduler(lr_max=args.lr, start_epoch=0, tot_epochs=args.epochs, 
                                  num_batches=len(train_loader), opt=optimizer, 
                                  div_factor=args.div_factor, pct_start=args.pct_start)
    
    scheduler.on_train_begin(epoch=0, n_epochs=args.epochs)    
    scheduler.jump_to_epoch(args.start_epoch)        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, linear_model, criterion, optimizer, epoch, scheduler, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, linear_model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'args': vars(args),
                'epoch': epoch + 1,
                'arch': args.arch,
                'layer_name': args.layer_name,
                'state_dict': linear_model.state_dict(),
                'best_acc1': best_acc1,
                'top1': acc1,
                'optimizer' : optimizer.state_dict(),
            }, filename=args.resume)
            
            #if epoch == args.start_epoch:
            #  sanity_check(linear_model.state_dict(), args.resume)


def train(train_loader, model, linear_model, criterion, optimizer, epoch, scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lr = AverageMeter('LR', ':6.6f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, lr],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    linear_model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # make sure we're not overheading our gpus
# This slows everything down. 
#         while not is_temp_safe(): pass
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # extract features from model
        with torch.no_grad():
            x = model(images)
            if args.cut is None:
                x = model.retained_layer(args.layer_name)
            x = x.detach()
        
        # pass features into linear model
        output = linear_model(x)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        lr.update(optimizer.param_groups[0]["lr"])
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        scheduler.on_batch_end(True)

def validate(val_loader, model, linear_model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    linear_model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            x = model(images)
            if args.cut is None:
                x = model.retained_layer(args.layer_name)
            output = linear_model(x)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    #if is_best:
    #    shutil.copyfile(filename, filename.replace('.pth.tar', '_best.pth.tar')

def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            
        return res


if __name__ == '__main__':
    main()
