import functools
import torch
from model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import numpy as np
import timm
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize, CenterCrop
)
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from brainscore_vision.model_helpers.s3 import load_weight_file

image_resize = 247
image_crop = 234
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


def custom_image_preprocess(images, **kwargs):
    transforms_val = Compose([
        Resize(image_resize, image_resize),
        CenterCrop(image_crop, image_crop),
        Normalize(mean=norm_mean, std=norm_std, ),
        ToTensorV2()])
    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)
    return images


def load_preprocess_images_custom(image_filepaths, preprocess_images=custom_image_preprocess, **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


class EffNetBX(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.efnet_model = timm.create_model('resnet50', pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x


def get_model_list():
    return ['resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234']


def get_model(name):
    assert name == 'resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234'
    model = EffNetBX()
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234/weights.pth",
                                    version_id="8t82bUmqChPEMT_q._hZ546E3ywhB2_S",
                                    sha1="f27b155eb492e0fb7b909bfe7e2a9e9bb295aa6f")
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))["model"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # strip `module.` prefix
        if "efnet" in k and not "attacker" in k:
            name = k.replace("module.model.", "")
            new_state_dict[name] = v
    model = model.efnet_model
    filter_elems = {"se", "act", "bn", "conv"}
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    preprocessing = functools.partial(load_preprocess_images_custom, preprocess_images=custom_image_preprocess)
    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing, batch_size=8)
    wrapper.image_size = image_crop
    return wrapper


def get_layers(name):
    assert name == 'resnet50_finetune_cutmix_e3_robust_linf8255_e0_247x234'
    return ['conv1',
            'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1',
            'layer2.1', 'layer2.2', 'layer2.3', 'layer3', 'layer3.0', 'layer3.0.downsample',
            'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4',
            'layer3.5',
            'layer4', 'layer4.0', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1',
            'layer4.2', 'global_pool', 'global_pool.flatten', 'global_pool.pool']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""