# Directly replicate ipcl primary models
#
# CUDA_VISIBLE_DEVICES=1 python step_by_step_imagenet_instance06_replicate_stack.py
# CUDA_VISIBLE_DEVICES=0 python step_by_step_imagenet_instance06_replicate_stack.py --data /n/holyscratch01/alvarez_lab/Lab/datasets/imagenet-256
#
#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from IPython.core.debugger import set_trace

import torchvision
import torchvision.transforms as transforms
import lib2.custom_transforms as custom_transforms

import os
import argparse
import time

# edit
# import models
import models.alexnet_half as models
import datasets
import math

# from lib2.NCEAverage import NCEAverage
# from lib2.NCEAverageUnlabeledMemMapStore import NCEAverage
# from lib2.NCEAverageUnlabeledMemMapStore2 import NCEAverage
# from lib2.NCEAverageUnlabeledMemMapStore3 import NCEAverage
from lib2.NCEAverageStack import NCEAverage
from lib2.LinearAverage import LinearAverage
from lib2.NCECriterion import NCECriterion
from lib2.utils import AverageMeter
from test_instance import NN, kNN, kNN_NN

# import horovod.torch as hvd
# hvd.init()


# In[4]:


from pathlib import Path
from addict import Dict

args = Dict({
    "data": '/home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256',
    "out_dir": "./results/step_by_step_imagenet_instance06_replicate_stack/ipcl_alpha_stack_alexnet_gn_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_rep2",
    "memory_device": 'gpu',
    # "memmap_filename": "/home/jovyan/work/Projects/InstanceNetReplication/lemniscate/results/instance_memmap_bm1_n5_lr03_t30/indexed_store.npy",
    "memmap_filename": '',
    "loader": "pt",
    "lr": 0.03,
    "cyclic": False,
    # "resume": './results/06_instance_imagenet_AlexNet_n5_lr03_pct40_t10_div1000_e100_bs128_bm20/05_instance_imagenet_AlexNet_n5_checkpoint.pth.tar',
    "test_only": False,
    "arch": "alexnet_half",
    "to_half": False,
    "low_dim": 128, # number of features in final layer
    "nce_k": 4096, # size of memory bank / number of nce samples
    "nce_t": 0.07, # softmax temperature parameter (origional = .1 for cifar10)
    "nce_m": 0.0, # weight momentum (original = 0.5)
    "nce_n": 5, # number of samples per image
    "epochs": 100,
    "include_self": True,
    "store_mean": False,
    "batch_size": 128,
    "batch_multiplier": 20,
    "pct_start": .40,
    "div_factor": 1000.0, 
    "print_freq": 500,
    "num_workers": 20,
    "max_prefetch": 0,
    "supervised": False,
    "groupnorm": True,
})

args.resume = os.path.join(args.out_dir, Path(args.out_dir).name + '_checkpoint.pth.tar')

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
    
parser = argparse.ArgumentParser(description='IPCL PyTorch Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset',
                    default='/home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256')
args_ = parser.parse_args()
args.data = args_.data

print(args)

visible_devices = [int(device) for device in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# In[5]:


import pandas as pd
import datasets
import numpy as np
# EDIT
# from lib.prefetch_new import PrefetchDataLoader
from lib2.prefetch import PrefetchDataLoader

if os.path.isfile(os.path.join(args.out_dir, 'results_by_epoch.csv')):
    df_epoch = pd.read_csv(os.path.join(args.out_dir, 'results_by_epoch.csv'))
else:
    df_epoch = pd.DataFrame(columns=['epoch', 'learning_rate', 'loss', 'accuracy'])
df_epoch


# In[6]:


# Results
df_iter = pd.DataFrame(columns=['epoch', 'iteration', 'learning_rate', 'loss'])
iter_no = 0
    
# Data
print('==> Preparing data...')
from loaders.imagenet.standard_samples import get_standard_imagenet as get_dataloaders

trainloader, recomputeloader, testloader = get_dataloaders(args.data, 
                                                           num_workers=args.num_workers,
                                                           batch_size=args.batch_size, 
                                                           n_samples=args.nce_n)

trainloader = PrefetchDataLoader(trainloader, max_prefetch=args.max_prefetch)
testloader = PrefetchDataLoader(testloader, max_prefetch=args.max_prefetch)

ndata = trainloader.dataset.__len__()
print('==> ready...')


# In[7]:


print('==> Building model..')
# EDIT
# net = models.__dict__[args.arch](in_channel=3, feat_dim=args.low_dim, to_half=args.to_half)
net = models.__dict__[args.arch](in_channel=3, low_dim=args.low_dim, to_half=args.to_half)

# In[8]:


from models.resnet import BasicBlock

print('==> Converting model to use groupnorm layers')

def convert_to_groupnorm(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.GroupNorm(32,child.num_features))
        elif isinstance(child, (nn.Sequential, BasicBlock)):
            convert_to_groupnorm(child)
    return model

if args.groupnorm: net = convert_to_groupnorm(net)
net
print(net)

# In[9]:


print('==> Initializing Memory Store')

# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m, 
                            memory_device=args.memory_device, memmap_filename=args.memmap_filename)
else:
    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()
    
# optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

net.to(device)
lemniscate.to(device)
criterion.to(device)


# In[10]:


# Model
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        del lemniscate
        torch.cuda.empty_cache()
        lemniscate = checkpoint['lemniscate']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    sys.exit(0)

if args.cyclic:
    scheduler = CosineWithRestarts(optimizer, len(trainloader), eta_min=0.00001, last_epoch=start_epoch-1, factor=2)
else: 
    scheduler = None    


# In[11]:


from lib2.one_cycle_scheduler import OneCycleScheduler

# scheduler = None
scheduler = OneCycleScheduler(lr_max=args.lr, start_epoch=0, tot_epochs=args.epochs, num_batches=len(trainloader), opt=optimizer, 
                              div_factor=args.div_factor, pct_start=args.pct_start)


# In[12]:


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr
#     if epoch >= 80:
#         lr = args.lr * (0.1 ** ((epoch-80) // 40))
#     print(lr)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def get_lr(epoch, cycle_len=5, lr_max=0.5, lr_min=.1):
    return lr_max - (epoch % cycle_len) * lr_min

def adjust_learning_rate(optimizer, epoch, lr_max):
    """https://towardsdatascience.com/faster-training-of-efficient-cnns-657953aa080"""
    lr_min = lr_max/25
    if epoch < 60:
        lr = get_lr(epoch+4, cycle_len=5, lr_max=lr_max, lr_min=lr_min)
    else:
        lr = get_lr(epoch, cycle_len=60, lr_max=lr_min, lr_min=lr_min/60)
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return
        
# Training
def train(epoch, iter_no, df_iter, batch_multiplier=10, print_freq=100, scheduler=None):  
    
    print('\nEpoch: %d' % epoch)
    if args.cyclic==False and scheduler is None:
        adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        if args.cyclic: scheduler.step()
        iter_no += 1
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        
        if batch_idx % batch_multiplier == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes) / float(batch_multiplier)

        loss.backward()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 or batch_idx == len(trainloader):
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'LR: {lr}'.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))

        df_iter = df_iter.append({
            "epoch": epoch, 
            "iteration": iter_no, 
            "learning_rate": optimizer.param_groups[0]['lr'], 
            "loss": train_loss.val, 
        }, ignore_index=True)  

        if scheduler is not None: scheduler.on_batch_end(True)
                    
    return iter_no, train_loss.avg, df_iter

# def train(epoch, iter_no, df_iter, print_freq=100, scheduler=None):    
    
#     print('\nEpoch: %d' % epoch)
#     if args.cyclic==False and scheduler is None:
#         adjust_learning_rate(optimizer, epoch, args.lr)
#     train_loss = AverageMeter()
#     data_time = AverageMeter()
#     batch_time = AverageMeter()
#     correct = 0
#     total = 0

#     # switch to train mode
#     net.train()

#     end = time.time()
#     for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
#         if args.cyclic: scheduler.step()
#         iter_no += 1
#         data_time.update(time.time() - end)
#         inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
#         optimizer.zero_grad()

#         features = net(inputs)
#         outputs = lemniscate(features, indexes)
#         loss = criterion(outputs, indexes)

#         loss.backward()
#         optimizer.step()

#         train_loss.update(loss.item(), inputs.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if batch_idx % print_freq == 0 or batch_idx == len(trainloader):
#             print('Epoch: [{}][{}/{}]'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
#                   'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
#                   'LR: {lr}'.format(
#                   epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))

#         df_iter = df_iter.append({
#             "epoch": epoch, 
#             "iteration": iter_no, 
#             "learning_rate": optimizer.param_groups[0]['lr'], 
#             "loss": train_loss.val, 
#         }, ignore_index=True)  
        
#         if scheduler is not None: scheduler.on_batch_end(True)
#     return iter_no, train_loss.avg, df_iter


# ## Find Best Learning Rate

# In[13]:


# from lib2.lr_find_samples import lr_warmup, lr_find, lr_plot
# import matplotlib.pyplot as plt

# stop_after_epoch = math.ceil(args.nce_k / (args.batch_size*args.nce_n))
# lr_warmup(net, trainloader, lemniscate, criterion, optimizer, stop_after_epoch=stop_after_epoch)
# lrs, log_lrs, losses = lr_find(net, trainloader, lemniscate, criterion, optimizer, num_it=310, start_lr=1e-04, end_lr=100000)
# lr_plot(lrs, losses, thresh=3e-01)


# In[14]:


# print(3e-01)
# lr_plot(lrs, losses, thresh=3e-01)


# ## Train Model

# In[15]:


from pathlib import Path
import shutil
import psutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    
    # make sure output directory exists:
    out_dir = Path(filename).parent
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print("=> saving checkpoint: {} \n".format(filename))
    
    # save checkpoint
    torch.save(state, filename)
  
    # save copy for best model
    if is_best:
        _filename = filename.replace("checkpoint.pth.tar", "best.pth.tar")
        print("=> saving best: {} \n".format(_filename))
        shutil.copyfile(filename, _filename)


# In[16]:


# acc = 0
# epoch = 0
# state = {
#         'epoch': epoch + 1,
#         'arch': args.arch,
#         'supervised': args.supervised,
#         'state_dict': net.state_dict(),
#         'lemniscate': lemniscate,
#         'prec1': acc,
#         'best_prec1': best_acc,            
#         'optimizer' : optimizer.state_dict(),
#     }
# state.keys()


# In[ ]:


import time
checkpoint_dir = os.path.join(args.out_dir, 'checkpoint')
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

start = time.time()
if scheduler is not None: scheduler.on_train_begin(epoch=0, n_epochs=args.epochs)    
if scheduler is not None: scheduler.jump_to_epoch(start_epoch)     
epoch = None
for epoch in range(start_epoch, args.epochs):
    iter_no, loss, df_iter = train(epoch, iter_no, df_iter, batch_multiplier=args.batch_multiplier, scheduler=scheduler, print_freq=args.print_freq)
    acc, top5, total = kNN(epoch, net, lemniscate, recomputeloader, testloader, 200, args.nce_t, recompute_memory=0)
    
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'supervised': args.supervised,
        'state_dict': net.state_dict(),
        'lemniscate': lemniscate,
        'prec1': acc,
        'best_prec1': best_acc,            
        'optimizer' : optimizer.state_dict(),
    }, is_best, args.resume)
    
    df_epoch = df_epoch.append({
        "epoch": epoch, 
        "loader": args.loader,
        "learning_rate": optimizer.param_groups[0]['lr'], 
        "loss": loss,
        "accuracy": acc,
    }, ignore_index=True)
    df_epoch.to_csv(os.path.join(args.out_dir, 'results_by_epoch.csv'), index=False)
    print('best accuracy: {:.2f}'.format(best_acc*100))
dur = time.time() - start
epoch = args.epochs if epoch is None else epoch
print("Total Time (min): {}".format(dur/60))


# In[ ]:


recomputeloader = PrefetchDataLoader(recomputeloader, max_prefetch=args.max_prefetch)
acc, top5, total = kNN(0, net, lemniscate, recomputeloader, testloader, 200, args.nce_t, 1)
print('last accuracy: {:.2f}'.format(acc*100))


# In[ ]:


print('last accuracy: {:.2f}'.format(acc*100))
df_epoch = df_epoch.append({
    "epoch": epoch + 1, 
    "loader": args.loader,
    "learning_rate": optimizer.param_groups[0]['lr'], 
    "loss": loss,
    "accuracy": acc,
}, ignore_index=True)

if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)    
df_epoch.to_csv(os.path.join(args.out_dir, 'results_by_epoch.csv'), index=False)


# In[ ]:


print('Saving final model')
save_checkpoint({
    'epoch': epoch + 1,
    'arch': args.arch,
    'supervised': args.supervised,
    'state_dict': net.state_dict(),
    'lemniscate': lemniscate,
    'top1': acc,
    'top5': top5,            
    'optimizer' : optimizer.state_dict(),
}, False, args.resume.replace("checkpoint.pth.tar", "final.pth.tar"))


# In[ ]:


# shutdown dataloaders, since their prefetching threads are running
print("=> shutting down dataloaders")
if isinstance(trainloader, PrefetchDataLoader): trainloader.shutdown()
if isinstance(recomputeloader, PrefetchDataLoader): recomputeloader.shutdown()
if isinstance(testloader, PrefetchDataLoader): testloader.shutdown()


# In[ ]:





# In[ ]:




