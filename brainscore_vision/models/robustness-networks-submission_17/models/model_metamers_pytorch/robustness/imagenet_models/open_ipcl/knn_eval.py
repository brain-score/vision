'''    
    
    doKNN

    Perform kNN readout.
    
    Usage:
    CUDA_VISIBLE_DEVICES='0' python knn_eval.py ipcl1 l2norm --data_dir /content/drive/MyDrive/datasets/imagenet-256
    CUDA_VISIBLE_DEVICES='1' python knn_eval.py ipcl1 l2norm --data_dir /home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256
 
'''
import os
import torch
import numpy as np
import pandas as pd
import inspect
from pprint import pprint
from glob import glob
from natsort import natsorted
from pathlib import Path
from IPython.core.debugger import set_trace
from copy import deepcopy
from itertools import combinations
from torchvision import transforms
import scipy.io as sio

from addict import Dict
def missing(self, key):
    raise KeyError(key)
Dict.__missing__ = missing

try:
    from fastprogress.fastprogress import master_bar, progress_bar
except:
    from fastprogress import master_bar, progress_bar
    
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from torch.utils.data import DataLoader
import models

from lib.knn import run_kNN_chunky as run_kNN

from fastscript import *

import torchvision.datasets as datasets

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # set_trace()
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
def get_dataloaders(root_dir, model, transform, dataset, batch_size=256, num_workers=16):        
        
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    # ====================================
    #  transforms
    # ====================================
    
    print(transform)
    
    # ====================================
    #  train_loader
    # ====================================
    
    print("==> training loader")
    train_dataset = ImageFolderInstance(train_dir, transform=transform)
    print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers, pin_memory=True)

    # ====================================
    #  val_loader (w/ train transforms, not for testing)
    # ====================================
    
    print("==> validation loader")
    val_dataset = ImageFolderInstance(val_dir, transform=transform)
    print(val_dataset)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def doKNN(OPTS):
    '''use kNN to perform classification based on features extracted from model'''
    OPTS = Dict(OPTS)                
    OPTS.func_name = inspect.stack()[0][3]
    
    # load model
    model, transform = models.__dict__[OPTS.model_name]()
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transform.transforms[-1] # model-specific normalization stats
    ])
    model.to(OPTS.device)
    print(model)
    
    # setup dataloaders
    train_loader, val_loader = get_dataloaders(OPTS.data_dir, model, transform, OPTS.dataset)
    
    # run kNN test
    top1, top5 = run_kNN(model, train_loader, val_loader, layer_name=OPTS.layer_name, 
                         K=OPTS.K, sigma=OPTS.tau, num_chunks=OPTS.chunk_size, 
                         out_device=OPTS.knn_device)
    
    if OPTS.results_file:
        df = pd.DataFrame(columns=['model_name','layer_name','dataset',
                                   'K','tau','chunk_size','top1','top5'])
        df = df.append({
            "model_name": OPTS.model_name,
            "layer_name": OPTS.layer_name,
            "dataset": OPTS.dataset,
            "K": OPTS.K,
            "tau": OPTS.tau,
            "chunk_size": OPTS.chunk_size,
            "top1": top1,
            "top5": top5,
        }, ignore_index=True)
        
        df.to_csv(OPTS.results_file, index=False)
        
    print("==> All Done!")
    return

@call_parse
def main(model_name:Param("name of model", str),
         layer_name:Param("name of layer for readout", str),
         dataset:Param("name of image dataset", str)='imagenet',
         data_dir:Param("where to find the data", str)='/home/jovyan/work/DataSetsLocal/ImageSets/imagenet/ILSRC2012-Pytorch-Short256',
         device:Param("which device to use", str)=None,
         
         # knn
         K:Param("number of neighbors", int)=200,
         tau:Param("number of neighbors", float)=.07,
         chunk_size:Param("chunks of training set to process (handle memory issues)", float)=10,
         knn_device:Param("which device to for kNN", str)=None,
         
         # data saving
         results_file:Param("name of results file", str)=None,
         out_dir:Param("where to store the results", str)=os.path.join(root, 'results', 'readout', 'kNN'),
        ):    
    
    OPTS = Dict({
        "analysis": "knn_eval",
        "model_name": model_name,
        "layer_name": layer_name,
        "dataset": dataset,
        "data_dir": data_dir,
        "K": K,
        "tau": tau,
        "chunk_size": chunk_size,
        "knn_device": knn_device,
        "results_file": results_file,
        "out_dir": out_dir,
        "device": device
    })                
            
    if OPTS.device is None:
        OPTS.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if OPTS.results_file is None:
        filename = f'{OPTS.model_name}_{OPTS.layer_name}_{OPTS.dataset}_kNN.csv'
        OPTS.results_file = os.path.join(OPTS.out_dir, filename)

    if not os.path.exists(OPTS.out_dir):
        os.makedirs(OPTS.out_dir)            
    
    if os.path.isfile(OPTS.results_file):
        print(f"\n=>skipping (already exists): {OPTS.results_file}\n")
        sys.exit(0)
        
    print(f"\n==> {OPTS.analysis}")
    pprint(OPTS)
    
    doKNN(OPTS)            
    
    return 