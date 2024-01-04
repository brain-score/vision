import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
import torch as ch
import numpy as np
import random
from model_analysis_folders.all_model_info import IMAGENET_PATH, MODEL_BASE_PATH
import os 

def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
         'gabors_f', 
         'v1_output',
         'bottleneck',
         'relu1_fake_relu',
         'relu2_fake_relu',
         'relu3_fake_relu',
         'relu4_fake_relu',
         'fc0_relu_fake_relu',
         'fc1_relu_fake_relu',
         'final'
    ]
    
    # Set the seeds so that we are always testing the random seed draw
    ch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ds = datasets.ImageNet(IMAGENET_PATH)

    ckpt_path = os.path.join(MODEL_BASE_PATH, 'visual_networks', 'pytorch_checkpoints', 'gvonealexnet_std4.pt')
    # NOTE: This model was trained with architecture `vonealexnet_gaussian_noise_std4` but 
    # for evaluation we load the "fixed" noise version. 
    model, _ = make_and_restore_model(arch='vonealexnet_gaussian_fixed_noise_std4', dataset=ds, 
                                      resume_path=ckpt_path,
                                      pytorch_pretrained=False, parallel=False,
                                     )

    # send the model to the GPU and return it. 
#     model.cuda()
    model.eval()
    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(return_metamer_layers=False,
         ds_kwargs={}):
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
