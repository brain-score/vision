import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
import robustness.data_augmentation as da
from torchvision import transforms
import torch
import numpy as np
from model_analysis_folders.all_model_info import IMAGENET_PATH, MODEL_BASE_PATH

def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    metamer_layers = [
         'input_after_preproc',
         's1_out',
         'c1_out',
         's2_out',
#          'c1_out_no_reshape',
#          's2_out_reshape_to_list',
         'c2_out',
         'final', 
    ]

    transform_hmax_train=transforms.Compose([
        transforms.RandomResizedCrop(250),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        da.Lighting(0.05, da.IMAGENET_PCA['eigval'],
                      da.IMAGENET_PCA['eigvec']),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    transform_hmax_test=transforms.Compose([
        transforms.Resize(250), # Using the transforms for metamer generation and do not want to crop image. 
        transforms.CenterCrop(250),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    # Better optimize the convolutations for this machine
    torch.backends.cudnn.benchmark=True

    ds = datasets.ImageNet(IMAGENET_PATH,
                          mean=[0.0], std=[1.0], max_value=255, min_value=0,
                          aug_train=transform_hmax_train,
                          aug_test=transform_hmax_test)
    ds.scale_image_save_PIL_factor = 1 # Do not scale the output images by 255 when saving with PIL
    ds.init_noise_mean = 255/2

    ckpt_path = os.path.join(MODEL_BASE_PATH, 'visual_networks', 'pytorch_checkpoints', 'hmax_linear_classification.pt')
    model, _ = make_and_restore_model(arch='hmax_standard_with_readout', dataset=ds, 
                                      resume_path=ckpt_path, 
                                      pytorch_pretrained=False, parallel=False)

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
