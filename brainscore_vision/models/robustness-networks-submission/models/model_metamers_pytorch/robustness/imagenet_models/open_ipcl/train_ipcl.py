#!/usr/bin/env python
# https://github.com/facebookresearch/moco
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as pth_transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import madgrad 
from lib.utils import madgrad_wd
# from torchlars import LARS
from fastprogress.fastprogress import master_bar, progress_bar 
from pdb import set_trace
from functools import partial
from pathlib import Path
from pprint import pprint

import models
from datasets.folder import ImageFolderInstanceSamples
from ipcl import IPCL0 as IPCL
from lib.knn_monitor import knn_monitor, run_kNN

from datasets import ImageFolderInstance, ImageFolderInstanceSamples
from kornia.filters import GaussianBlur2d as GaussianBlurGPU
from albumentations import (
    SmallestMaxSize, RandomResizedCrop, HorizontalFlip, CenterCrop,
    Compose as AlbumCompose,
)
from dataloaders.transforms import (
    InstanceSamplesTransform, ToNumpy,
    ToChannelsFirst, ToDevice, ToFloatDiv,
    ColorJitterGPU, HSVJitterGPU, RandomContrastGPU, RandomGrayscaleGPU, RandomGaussianBlurGPU,
    NormalizeGPU, Compose, RandomApply, RandomRotateGPU,
    CircularMaskGPU, FixedOpticalDistortionGPU,
    SRGB_to_LMS, LMS_To_LGN, LMS_To_LGN_Lum, LMS_To_LGN_Color,
    CenterCropResize
)

from dataloaders.utils import open_image, open_image_array
from dataloaders.dataloader import FastLoader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='IPCL PyTorch Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet_gn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet_gn_cifar10)')
parser.add_argument('-j', '--num_workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('--num_epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', default=0, type=int,
                    help='run evaluation only')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

# optimizer configs
parser.add_argument('--opt', default='MADGRAD', type=str,
                    help='which optimizer to use')
parser.add_argument('--use-lars', default=0, type=int,
                    help='whether to use LARS optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
parser.add_argument('--bm', '--batch-multiplier', default=20.0, type=float,
                    metavar='BM', help='batch multiplier', dest='batch_multiplier')
parser.add_argument('--scheduler', default=None,
                    choices=['','CosineAnnealing','OneCycle'],
                    help='learning rate scheduler: ' +
                        ' | '.join(['CosineAnnealing','OneCycle']) +
                        ' (default: OneCycle)')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--div_factor', default=1000.0, type=float, 
                    help='sets initial lr for onecycle (lr/div_factor)')
parser.add_argument('--pct_start', default=.40, type=float, 
                    help='pct of epochs rise to max lr (default = .40)')
parser.add_argument('--tau_scheduler', default=None,
                    choices=['CosineAnnealing','OneCycle'],
                    help='tau scheduler: ' +
                        ' | '.join(['CosineAnnealing','OneCycle']) +
                        ' (default: None)')

# ipcl specific configs:
parser.add_argument('--ipcl-dim', default=128, type=int,
                    help='output dimension (default: 128)')
parser.add_argument('--ipcl-k', default=4096, type=int,
                    help='queue size; number of negative samples retained (default: 4096)')
parser.add_argument('--ipcl-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--ipcl-n', default=5, type=int,
                    help='number of augmentations per image, used to compute prototype (default: 5)')

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class AlbumTransforms(object):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return self.transform(image=x)['image']
        
def get_transforms_custom(image_size=256, crop_size=224, n_samples=5, device=None,
                          do_blur=False, rotation=0, pad_mode='zeros', 
                          circular_window=False, distortion=False,
                          mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        
    # =========================
    #  Before Batch Transforms
    # =========================

    before_tranforms = [
        SmallestMaxSize(image_size),
        RandomResizedCrop(crop_size,crop_size,scale=(0.2, 1.), ratio=(0.75, 1.3333333333333333), p=1.0),
        HorizontalFlip(p=.5),
    ]

    tfrm = AlbumCompose(before_tranforms)
    print(tfrm)
    
    #before_batch = InstanceSamplesTransform(pth_transforms.Compose([lambda x: tfrm(image=x)['image']]), 
    #                                        n_samples=n_samples)          
    
    # before_batch = pth_transforms.Compose([lambda x: tfrm(image=x)['image']]) 
    before_batch = pth_transforms.Compose([AlbumTransforms(tfrm)])
    
    # =========================
    #  After Batch Transforms
    # =========================

    # similar to wu et al
    baseline_transforms = [
        ToChannelsFirst(),
        ToDevice(device),
        ToFloatDiv(255.),  
        RandomGrayscaleGPU(p=0.2),
        ColorJitterGPU(p=1.0, hue=.4, saturation=.4, value=.4, contrast=.4),
    ]
    
    after_transforms = baseline_transforms
        
    if do_blur:
        after_transforms += [RandomGaussianBlurGPU(p=0.5, sigma=[.1, 2.])]
    
    if rotation:
        after_transforms += [RandomRotateGPU(p=1.0, max_deg=rotation, pad_mode=pad_mode)]
    
    if circular_window:
        after_transforms += [CircularMaskGPU(crop_size, blur_span=12, device=device)]
        
    if distortion:
        after_transforms += [FixedOpticalDistortionGPU(crop_size, crop_size, device=device)]        
    
    if mean and std:
        after_transforms += [NormalizeGPU(mean=mean, std=std, device=device)]
        
    after_batch_tfrm = pth_transforms.Compose(after_transforms)

    print(after_batch_tfrm)

    def after_batch(batch):
        if isinstance(batch[0], list):
            batch[0] = [after_batch_tfrm(b) for b in batch[0]]
        else:
            batch[0] = after_batch_tfrm(batch[0])
        return batch

    return before_batch, after_batch

def main():
    args = parser.parse_args()

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.gpu = gpu
    args.device = f'cuda:{args.gpu}' if args.gpu is not None else device
    
    # default filename
    if args.resume == '':
        filename = 'ipcl0_{}_{}_{}_lars{}_bs{}_bm_{}_ep{}_out{}_k{}_n{}_t{}_lr{}.pth.tar'.format(
            args.arch,
            args.opt,
            args.scheduler,
            args.use_lars,
            args.batch_size,
            args.batch_multiplier,
            args.num_epochs,
            args.ipcl_dim,
            args.ipcl_k,
            args.ipcl_n,
            args.ipcl_t,
            args.lr)
        
        args.resume = os.path.join('results', filename)
      
    if args.scheduler == '': args.scheduler = None
        
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
   
    # ----------------------------------------------
    #  INIT IPCL MODEL
    # ----------------------------------------------
    
    print("=> creating model '{}'".format(args.arch))
    model = IPCL(models.__dict__[args.arch](out_dim=args.ipcl_dim), 
                 1281167, # number of imagenet images
                 K=args.ipcl_k, 
                 T=args.ipcl_t, 
                 out_dim=args.ipcl_dim, 
                 n_samples=args.ipcl_n)  
    print(model)
    
    # ----------------------------------------------
    #  INIT OPTIMIZER
    # ----------------------------------------------
    
    if args.opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.wd)
    elif args.opt == "MADGRAD":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                                    weight_decay=args.wd, eps=1e-06)
    elif args.opt == "madgrad_wd":
        optimizer = madgrad_wd(model.parameters(), lr=args.lr, momentum=args.momentum, 
                               weight_decay=args.wd, eps=1e-06)
            
    if args.use_lars:
        #torchlars did not work with our gradient accumulation (batch multiplier) scheme
        #from torchlars import LARS
        #optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
        from utils import LARS
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, eta=0.001)
        
        
    print(optimizer)
    
    # ----------------------------------------------
    #  DATA LOADERS
    # ----------------------------------------------
    
    # TRANSFORMS
    root_dir = Path(args.data)
    before_batch_train_tfrm, after_batch_train_trfm = get_transforms_custom(n_samples=args.ipcl_n)
    
    # DATALOADERS for IPCL training / validation    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    # TODO: modify instance sampling to happen before collation (as a transform)
    # so that we don't have to use torch.cat to stack imgs (which creates a new copy, like slowing down)
    train_dataset = ImageFolderInstanceSamples(root=root_dir/"train", 
                                               n_samples=args.ipcl_n,
                                               loader=open_image_array, 
                                               transform=before_batch_train_tfrm)
        
    val_dataset = ImageFolderInstanceSamples(root=root_dir/"val", 
                                             n_samples=args.ipcl_n,
                                             loader=open_image_array, 
                                             transform=before_batch_train_tfrm)
    
    assert len(train_dataset) == 1281167, f"Oops, expected num train images = 1281167, got {len(train_dataset)}"
    assert len(val_dataset) == 50000, f"Oops, expected num train images = 50000, got {len(val_dataset)}"
    
    print(train_dataset)
    print(val_dataset)
    
    train_loader = FastLoader(
        train_dataset, 
        after_batch=after_batch_train_trfm, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.num_workers, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=False)
        
    val_loader = FastLoader(
        val_dataset, 
        after_batch=after_batch_train_trfm, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=False)

    # DATALOADERS for KNN monitoring
    knn_transform = pth_transforms.Compose([
        pth_transforms.Resize(256),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
        
    train_dataset = ImageFolderInstanceSamples(
        root=root_dir/"train", n_samples=1, transform=knn_transform
    )
    print(train_dataset)

    val_dataset = ImageFolderInstanceSamples(
        root=root_dir/"val", n_samples=1, transform=knn_transform
    )
    print(val_dataset)

    train_loader_knn = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    val_loader_knn = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    loaders = [train_loader, val_loader, train_loader_knn, val_loader_knn]
        
    # ----------------------------------------------
    #  SCHEDULERS
    # ----------------------------------------------
    
    epoch_scheduler, batch_scheduler, tau_scheduler = None, None, None 
    if args.scheduler == "CosineAnnealing":
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                     T_max=args.num_epochs)
    elif args.scheduler == "OneCycle":
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                              max_lr=args.lr, 
                                                              div_factor=args.div_factor,
                                                              final_div_factor=1000*1e4,
                                                              pct_start=args.pct_start,
                                                              anneal_strategy='cos',
                                                              cycle_momentum=True,
                                                              base_momentum=0.85,
                                                              max_momentum=0.95,
                                                              steps_per_epoch=len(loaders[0]), 
                                                              epochs=args.num_epochs)
    elif args.scheduler == "CosineAnnealingWarmup":
        batch_scheduler = adjust_learning_rate

    if args.tau_scheduler is not None:
        print("==> setting up tau scheduler:")
        tau_scheduler = partial(cosine_annealing, total_iters=args.num_epochs*len(loaders[0]), 
                                start_val=args.tau_scheduler.start_val, 
                                end_val=args.tau_scheduler.end_val)
        
    # ----------------------------------------------
    #  DISTRIBUTED TRAINING
    # ----------------------------------------------
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # ----------------------------------------------
    #  RESUME FROM CHECKPOINT
    # ----------------------------------------------
    
    start_epoch = 0
    perf_monitor = None        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if batch_scheduler is not None:
                batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
            if epoch_scheduler is not None:
                epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
            perf_monitor = checkpoint['perf_monitor']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
    elif args.resume is not None:
        print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> no checkpoint given, skipping saving...")
        
    cudnn.benchmark = True
    
    # ----------------------------------------------
    #  EVALUATE
    # ---------------------------------------------- 
    
    if args.evaluate:
        run_kNN(model.base_encoder, train_loader_knn, val_loader_knn, knn_device=args.device)
        return
    
    # ----------------------------------------------
    #  TRAIN
    # ----------------------------------------------        
    
    train_model(model, 
                optimizer, 
                loaders, 
                start_epoch, 
                args.num_epochs,
                train_sampler=train_sampler,
                epoch_scheduler=epoch_scheduler,
                batch_scheduler=batch_scheduler,
                tau_scheduler=tau_scheduler,
                perf_monitor=perf_monitor,
                args=args)

def train_model(
    learner,
    optimizer,
    loaders,
    start_epoch,
    num_epochs,
    train_sampler=None,
    epoch_scheduler=None,
    batch_scheduler=None,
    tau_scheduler=None,
    perf_monitor=None,
    args={},
):
    """
        Typically models output activations or logits.

        A `learner` is a wrapper that includes all the logic needed to compute
        the loss, and returns the loss instead.

        This is helpful when your model architecture has multiple components,
        and more complex logic for computing the loss, as is often the case
        with self-supervised systems (e.g., IPCL, MoCo2, BYOL, SimCLR).

    """
    print(f"=> Training models {args.arch}")
    pprint(args)
    
    train_loader, val_loader, train_loader_knn, val_loader_knn = loaders

    mb = master_bar(range(start_epoch, num_epochs))
    mb.names = ["top1"]
    if perf_monitor is None:
        perf_monitor = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "top1": [],
        }
        
    for epoch in mb:
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, trainX, trainY, _ = train(
            epoch,
            learner,
            optimizer,
            train_loader,
            args,
            batch_multiplier=args.batch_multiplier,
            scheduler=batch_scheduler,
            tau_scheduler=tau_scheduler,
            mb=mb
        )

        # val_loss = validate(learner, val_loader, args, mb=mb)
        top1, top5 = knn_monitor(
            learner.base_encoder, trainX, trainY, val_loader_knn, sigma=learner.T, knn_device=args.device
        )
 
        # track results
        perf_monitor["epoch"].append(epoch)
        perf_monitor["top1"].append(top1)
        perf_monitor["train_loss"].append(train_loss)
        #perf_monitor["val_loss"].append(val_loss)
        graphs = [[perf_monitor["epoch"], perf_monitor["top1"]]]
        x_bounds = [0, num_epochs]
        y_bounds = [0, max(perf_monitor["top1"]) * 1.1]
        mb.update_graph(graphs, x_bounds, y_bounds)

        # update the learning rate
        if epoch_scheduler is not None:
            epoch_scheduler.step()
                
        # save
        if args.resume is not None:
            torch.save({
            'args': args if isinstance(args, dict) else vars(args),
            'epoch': epoch + 1,
            'state_dict': learner.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'epoch_scheduler': epoch_scheduler.state_dict() if hasattr(epoch_scheduler, 'state_dict') else None,
            'batch_scheduler': batch_scheduler.state_dict() if hasattr(batch_scheduler, 'state_dict') else None,
            'top1': top1,
            'perf_monitor': perf_monitor,
            #'train_transform': train_loader.dataset.transform,
            #'val_transform': val_loader.dataset.transform,
          }, args.resume, _use_new_zipfile_serialization=False)

    top1, top5 = run_kNN(learner.base_encoder, train_loader_knn, val_loader_knn, knn_device=args.device)

    print("=> all done!")
            
            
def train(epoch, learner, optimizer, train_loader, args, batch_multiplier=1.,
          scheduler=None, tau_scheduler=None, mb=None):

    losses = AverageMeter('Train Loss', ':.4e')
    taus = AverageMeter('Tau', ':.4e')
  
    features,targets,indexes=[],[],[]

    learner.train() 
    optimizer.zero_grad()
    for batch_num,(inputs,targs,idxs) in enumerate(progress_bar(train_loader, parent=mb)):
        iter_num = epoch*len(train_loader) + batch_num
        
        if args.scheduler == "CosineAnnealingWarmup":
            lr = adjust_learning_rate(args, optimizer, train_loader, iter_num)
        
        if tau_scheduler is not None:
            learner.T = tau_scheduler(iter_num)
        taus.update(learner.T, 1)

        if isinstance(inputs, list):
            inputs = torch.cat(inputs).to(args.device, non_blocking=True)
        else:
            inputs = inputs.to(args.device, non_blocking=True)

        if isinstance(targs, list):
            targs = torch.cat(targs).to(args.device, non_blocking=True)
        else:
            targs = targs.to(args.device)

        if isinstance(idxs, list):
            idxs = torch.cat(idxs)        
    
        #optimizer.zero_grad()
        loss, (embeddings, prototypes) = learner(inputs, targs)
        loss = loss / float(batch_multiplier)
        losses.update(loss.item(), 1)
        loss.backward()
        #optimizer.step()
        if (iter_num+1) % batch_multiplier == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        # store embeddings for online knn monitoring
        # keep for only the first augmented instance
        feat = embeddings.chunk(args.ipcl_n)[0].detach().cpu()
        feat = feat.view(feat.shape[0],-1)
        features.append(feat)
        targets.append(targs.chunk(args.ipcl_n)[0].cpu())
        indexes.append(idxs.chunk(args.ipcl_n)[0].cpu())
    
        if batch_num % 250 == 0 or batch_num==(len(train_loader)-1):
            lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch}][{batch_num}/{len(train_loader)}]: LOSS: {losses.avg:6.3f}, LR: {lr}, TAU: {taus.avg:6.3f}") 

        if scheduler is not None and args.scheduler != "CosineAnnealingWarmup": 
            scheduler.step()

        #if batch_num==4: break
        
    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    indexes = torch.cat(indexes, dim=0)

    return losses.avg, features, targets, indexes

def validate(learner, val_loader, args, mb=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses = AverageMeter('Val Loss', ':.4e')

    learner.eval() 
    with torch.no_grad():
        for batch_num,(inputs,targs,indexes) in enumerate(progress_bar(val_loader, parent=mb)):
            if isinstance(inputs, list):
                inputs = torch.cat(inputs).to(args.device, non_blocking=True)
            else:
                inputs = inputs.to(args.device, non_blocking=True)
      
            if isinstance(targs, list):
                targs = torch.cat(targs).to(args.device, non_blocking=True)
            else:
                targs = targs.to(args.device)

            loss,_ = learner(inputs, targs)
            losses.update(loss.item(), 1)

            #if batch_num==4: break 

    print(f'=> Val Loss: {losses.avg:6.3f}\n')

    return losses.avg            

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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


def cosine_annealing(iter_num, total_iters, start_val, end_val):
    return end_val + (start_val - end_val) * (1 + math.cos(math.pi * iter_num / total_iters)) / 2

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.num_epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.lr * (args.batch_size * args.batch_multiplier) / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    '''
    
        CUDA_VISIBLE_DEVICES='0' python train_ipcl.py -a alexnet_gn --gpu 0 --opt SGD --scheduler OneCycle --use-lars 0 -b 256 --batch-multiplier 10 /n/holyscratch01/alvarez_lab/Lab/datasets/imagenet-256
        
        CUDA_VISIBLE_DEVICES='0' python train_ipcl.py -b 128 -bm 20 -a alexnet_gn --gpu 0 --opt SGD --scheduler OneCycle --use-lars 0 /n/holyscratch01/alvarez_lab/Lab/datasets/imagenet-256

    '''
    main()