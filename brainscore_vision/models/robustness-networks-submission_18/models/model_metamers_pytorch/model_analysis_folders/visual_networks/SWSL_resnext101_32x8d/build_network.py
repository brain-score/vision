"""Constructs a semi-weakly supervised ResNeXt-101 32x8 model pre-trained on 1B weakly supervised
   image dataset and finetuned on ImageNet.
   `"Billion-scale Semi-Supervised Learning for Image Classification" <https://arxiv.org/abs/1905.00546>`_
   Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
"""
import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
import torch
torch.backends.cudnn.benchmark = True

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
#          'conv1_relu1',
         'conv1_relu1_fake_relu',
#          'layer1',
         'layer1_fake_relu',
#          'layer2',
         'layer2_fake_relu',
#          'layer3',
         'layer3_fake_relu',
#          'layer4',
         'layer4_fake_relu',
         'globalpool',
         'final'
    ]

#     ds = datasets.ImageNet('/om2/data/public/imagenet/images_complete/ilsvrc/', mean=[0.0], std=[1.0],)
    ds = datasets.ImageNet('/om2/data/public/imagenet/images_complete/ilsvrc/')

    change_prefix_checkpoint = {}
    remap_checkpoint_keys = {}

    model, _ = make_and_restore_model(arch='swsl_resnext101_32x8d', dataset=ds,
                                      pytorch_pretrained=True, parallel=False, strict=False,
                                      remap_checkpoint_keys=remap_checkpoint_keys,
                                      change_prefix_checkpoint=change_prefix_checkpoint,
                                      append_name_front_keys=['module.model.', 'module.attacker.model.'])
    print(model)
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
