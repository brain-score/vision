import os
import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.imagenet_models.open_ipcl import models as ipcl_models
from robustness.imagenet_models.custom_modules import SequentialWithAllOutput
import torch 
from PIL import Image

import torch
torch.backends.cudnn.benchmark = True

# For the classifer layer
class LinearReadout(torch.nn.Module):
    def __init__(self, in_features, num_classes: int = 1000) -> None:
        super(LinearReadout, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.classifier = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, with_latent=False, no_relu=False, fake_relu=False) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if with_latent:
            all_outputs = {'final':x}
            return x, None, all_outputs
        return x

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
    model, transforms = ipcl_models.ipcl1(no_embedding=True)
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

    # Add on the classifier layer. 
    linear_ckpt_path = '/net/oms.ib.cluster/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/model_training_directory/imagenet_networks/ipcl_model_01_alexnet_fc7_ipcl_training_head/weights/ipcl1_fc7_lincls_onecycle.pth.tar'
    checkpoint_linear_classifier = torch.load(linear_ckpt_path)
    in_features = 4096
    linear_model = LinearReadout(in_features, num_classes=1000)
    linear_model.load_state_dict(checkpoint_linear_classifier['state_dict'])

    model = SequentialWithAllOutput(model, linear_model)
    
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
