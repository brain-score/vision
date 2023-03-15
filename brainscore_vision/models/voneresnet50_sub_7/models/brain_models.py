import functools

import torch
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from model_tools.brain_transformation import ModelCommitment
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize,CenterCrop
    )
from albumentations.pytorch import ToTensorV2
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models
import vonenet.vonenet

import os 

image_resize = 304
image_crop = 266
norm_mean = [0.5, 0.5, 0.5] 
norm_std =  [0.5, 0.5, 0.5]

layers = ['module.vone_block',
        'module.model.layer1.0', 'module.model.layer1.1', 'module.model.layer1.2',
        'module.model.layer2.0', 'module.model.layer2.1', 'module.model.layer2.2', 'module.model.layer2.3',
        'module.model.layer3.0', 'module.model.layer3.1', 'module.model.layer3.2', 'module.model.layer3.3',
        'module.model.layer3.4', 'module.model.layer3.5',
        'module.model.layer4.0', 'module.model.layer4.1', 'module.model.layer4.2',
        'module.model.avgpool']

def custom_image_preprocess(images, **kwargs):
    
    transforms_val = Compose([
        Resize(image_resize, image_resize),
        CenterCrop(image_crop, image_crop),
        Normalize(mean=norm_mean,std=norm_std,),
        ToTensorV2()])
    
    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)

    return images

def load_preprocess_images_custom(image_filepaths, preprocess_images=custom_image_preprocess,  **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images

def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper()\
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image




class VOneResNet50(nn.Module):
    def __init__(self):
        super().__init__ ()
        self.efnet_model = vonenet.get_model(model_arch="resnet50_ns", pretrained=True, map_location="cpu")
        #self.bottleneck = nn.Conv2d(512, 24, kernel_size=1, stride=1, bias=False)
        #nn.init.kaiming_normal_(self.bottleneck.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.efnet_model(x)
        
        return x, 0, 0

model_tf_efficientnet_b1_ns= VOneResNet50()
dir_path = os.path.dirname(os.path.realpath(__file__))
model_tf_efficientnet_b1_ns.load_state_dict(torch.load(dir_path + "/vone_resnet50_ns_cutmixpatchresize_augmix_epoch4_score0.5299826824628522_best.pth", map_location=torch.device('cpu'))["model"])
model = model_tf_efficientnet_b1_ns.efnet_model
filter_elems = set([])
layer_list = [layer for layer, _ in model.named_modules() if not any(i in layer for i in filter_elems)]
print(layer_list)
print(len(layer_list))


preprocessing = functools.partial(load_preprocess_images_custom, 
                                    preprocess_images=custom_image_preprocess,
                                    )


activations_model  = PytorchWrapper(identifier='vonern50_cutmixpatch_augmix_e4_304x266', model=model, preprocessing=preprocessing, batch_size=8)
model = ModelCommitment(identifier='vonern50_cutmixpatch_augmix_e4_304x266', activations_model=activations_model ,
                    # specify layers to consider
                    layers=layers)
model.layer_model.region_layer_map['V1'] = 'module.vone_block.output'

def get_model_list():
    return ['vonern50_cutmixpatch_augmix_e4_304x266']

def get_model(name):
    assert name == 'vonern50_cutmixpatch_augmix_e4_304x266'

    return model


def get_bibtex(model_identifier):
    return """
    @article {Dapello2020.06.16.154542,
	author = {Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David D. and DiCarlo, James J.},
	title = {Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations},
	elocation-id = {2020.06.16.154542},
	year = {2020},
	doi = {10.1101/2020.06.16.154542},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/10/22/2020.06.16.154542},
	eprint = {https://www.biorxiv.org/content/early/2020/10/22/2020.06.16.154542.full.pdf},
	journal = {bioRxiv}
                """


if __name__ == '__main__':
    check_models.check_brain_models(__name__)
