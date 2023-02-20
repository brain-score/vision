import functools

import torch
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize
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

image_size = 224
norm_mean = [0.485, 0.456, 0.406] 
norm_std = [0.229, 0.224, 0.225]

def custom_image_preprocess(images, **kwargs):
    
    transforms_val = Compose([
          #CenterCrop(256,256),
          Resize(image_size, image_size),
          Normalize(mean=norm_mean,std=norm_std),
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

def get_model_list():
    return ['effnetb1_vonemap_retrain_epoch4']


class EffNetVOneMaps(nn.Module):
    def __init__(self):
        super().__init__ ()
        self.vone_block = vonenet.get_model(model_arch=None, pretrained=False, stride=8, simple_channels=120, complex_channels=120)
        self.efnet_model = timm.create_model("tf_efficientnet_b1_ns", pretrained=True)

    def forward(self, x):
        x_vone = self.vone_block(x)
        #print("vone", x_vone.shape)

        x = self.efnet_model.conv_stem(x)
        x = self.efnet_model.bn1(x)
        x = self.efnet_model.act1(x)

        x = self.efnet_model.blocks[0](x)
        x = self.efnet_model.blocks[1](x)
        x = self.efnet_model.blocks[2](x)
        #x = self.efnet_model.blocks[3][0](x)

        blocks30conv_pw = self.efnet_model.blocks[3][0].conv_pw(x)
        #print("blocks30conv_pw", blocks30conv_pw.shape)
        x = self.efnet_model.blocks[3][0].bn1(blocks30conv_pw)
        x = self.efnet_model.blocks[3][0].act1(x)
        x = self.efnet_model.blocks[3][0].conv_dw(x)
        x = self.efnet_model.blocks[3][0].bn2(x)
        x = self.efnet_model.blocks[3][0].act2(x)
        x = self.efnet_model.blocks[3][0].se(x)
        x = self.efnet_model.blocks[3][0].conv_pwl(x)
        x = self.efnet_model.blocks[3][0].bn3(x)

        x = self.efnet_model.blocks[3][1](x)
        x = self.efnet_model.blocks[3][2](x)
        x = self.efnet_model.blocks[3][3](x)
        x = self.efnet_model.blocks[4](x)
        x = self.efnet_model.blocks[5](x)
        x = self.efnet_model.blocks[6](x)

        x = self.efnet_model.conv_head(x)
        x = self.efnet_model.bn2(x)
        x = self.efnet_model.act2(x)
        x = self.efnet_model.global_pool(x)
        if self.efnet_model.drop_rate > 0.:
            x = F.dropout(x, p=self.efnet_model.drop_rate, training=self.efnet_model.training)
        x = self.efnet_model.classifier(x)

        return x#, blocks30conv_pw, x_vone

def get_model(name):
    assert name == 'effnetb1_vonemap_retrain_epoch4'
    model= EffNetVOneMaps()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load(dir_path + "/effnetb1_vonemaps_epoch4_score0.4938056038073132_best.pth", map_location=torch.device('cpu'))["model"])

    for name, _ in model.named_modules():
        print(name)
    
    preprocessing = functools.partial(load_preprocess_images_custom, 
                                        preprocess_images=custom_image_preprocess,
                                        )


    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing, batch_size=8)

    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == 'effnetb1_vonemap_retrain_epoch4'
    return [
        "vone_block.module", "vone_block.module.gabors", "vone_block.module.simple", "vone_block.module.complex", 
        "efnet_model.blocks.1.0", "efnet_model.blocks.1.1","efnet_model.blocks.1.2",
        "efnet_model.blocks.2.0", "efnet_model.blocks.2.1", "efnet_model.blocks.2.2", 
        "efnet_model.blocks.3.0",
        #"efnet_model.blocks.3.0.conv_pw", "efnet_model.blocks.3.0.bn1", "efnet_model.blocks.3.0.conv_dw", "efnet_model.blocks.3.0.act1", 
        "efnet_model.blocks.3.1", "efnet_model.blocks.3.2", "efnet_model.blocks.3.3",
        "efnet_model.blocks.4.0", "efnet_model.blocks.4.1", "efnet_model.blocks.4.2", "efnet_model.blocks.4.3", 
        "efnet_model.blocks.5.0", "efnet_model.blocks.5.1", "efnet_model.blocks.5.2", "efnet_model.blocks.5.3", "efnet_model.blocks.5.4", 
        "efnet_model.blocks.6.0", "efnet_model.blocks.6.1",
        "efnet_model.global_pool.flatten", "efnet_model.global_pool.pool", "efnet_model.global_pool"]


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
