import os
import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.imagenet_models.open_ipcl import models as ipcl_models
import torch 
from PIL import Image

import torch
torch.backends.cudnn.benchmark = True

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(ds_kwargs={}, return_metamer_layers=False, dataset_name='ImageNet'):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
         'conv_block_1',
         'conv_block_2',
         'conv_block_3',
         'conv_block_4',
         'conv_block_5',
         'avgpool',
         'fc6', 
         'fc7',
         'final'
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transforms = ipcl_models.ipcl12()
    model.to(device)

    ds = datasets.ImageNet('/om2/data/public/imagenet/images_complete/ilsvrc/', # '/om2/data/public/imagenet/images_complete/ilsvrc/',
                       mean=model.config['mean'],
                       std=model.config['std'],
                       min_value = 0,
                       max_value = 1,
                       aug_train=transforms, # transforms,
                       aug_test=model.val_transform)# transforms)

    # These are required since we have a non-default mean. 
    # These are the correct values (defaults) if we have min=0 max=1. 
    ds.scale_image_save_PIL_factor = 255 # Do not scale the output images by 255 when saving with PIL
    ds.init_noise_mean = 0.5
    
    model = AttackerModel(model, ds)

    # send the model to the GPU and return it.
    model.cuda()
    model.eval()

    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(return_metamer_layers=False,
         ds_kwargs={}, dataset_name='ImageNet'):
    if return_metamer_layers: 
        model, ds, metamer_layers = build_net(
                                              return_metamer_layers=return_metamer_layers,
                                              ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(
                              return_metamer_layers=return_metamer_layers,
                              ds_kwargs=ds_kwargs)
        return model, ds


if __name__== "__main__":
    main()
