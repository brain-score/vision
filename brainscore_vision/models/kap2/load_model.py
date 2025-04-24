
import torch
import torch.nn as nn
import torchvision
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



from collections import namedtuple
from torch.utils.data import Dataset, DataLoader


#import spacetorch.analyses.rsa as rsa

import argparse
from urllib.request import urlretrieve


from .resnet_imagenet_continuoustopo import ResNet18


def detach(tensor):
    return tensor.detach().cpu().numpy()

def load_model(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    #model = get_model(args) #arch change
    model = ResNet18(1000, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
    
    
    url = f'https://brainscore-storage.s3.us-east-2.amazonaws.com/brainscore-vision/models/cln_resnet18contopo_gaussian_1_0.0gaussian_0.23_continuous_prog_t/resnet18contopo_100.pt'
    fh = urlretrieve(url, f'resnet18contopo_100.pt')
    load_path = fh[0]
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
    
    
    #state_dict = torch.load(f'resnet18contopo_100.pt')
    
    
    
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")

    
    return model
  
  
