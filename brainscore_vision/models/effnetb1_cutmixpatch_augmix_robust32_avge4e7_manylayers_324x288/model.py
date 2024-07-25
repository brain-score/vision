import functools
import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn as nn
from albumentations import (
    Compose, Normalize, Resize, CenterCrop
)
from albumentations.pytorch import ToTensorV2
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.s3 import load_weight_file


image_resize = 324
image_crop = 288
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
freeze_layers = ['blocks.0.0', 'blocks.0.1', 'blocks.1.0',
                 'blocks.1.1', 'blocks.1.2', 'blocks.2.0',
                 'blocks.2.1', 'blocks.2.2', 'blocks.3.0']


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
        self.efnet_model = timm.create_model('tf_efficientnet_b1_ns', pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x


def get_model(name):
    assert name == 'effnetb1_cutmixpatch_augmix_robust32_avge4e7_manylayers_324x288'
    model_tf_efficientnet_b1_ns = EffNetBX()
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="effnetb1_cutmixpatch_augmix_robust32_avge4e7_manylayers_324x288/tf_efficientnet_b1_ns_robust_cutmixpatchresize_augmix_e4toe7.pth",
                                    version_id="iB0UqbguDpYHD0HRbMt1F1er3c414yWr",
                                    sha1="37f3ac1b14e80cfaa99fa5f412c1e132480ed5b6")

    model_tf_efficientnet_b1_ns.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu'))["model"])
    model = model_tf_efficientnet_b1_ns.efnet_model
    filter_elems = set(["se", "act", "bn", "conv"])
    layer_list = [layer for layer, _ in model.named_modules() if not any(i in layer for i in filter_elems)]
    print(layer_list)
    print(len(layer_list))

    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and any(x in n for x in ["conv_stem"] + freeze_layers) or n == "bn1":
            print(f"Freeze {n, m}")
            m.eval()

    preprocessing = functools.partial(load_preprocess_images_custom,
                                      preprocess_images=custom_image_preprocess,
                                      )

    wrapper = PytorchWrapper(identifier='my-model', model=model, preprocessing=preprocessing, batch_size=8)

    wrapper.image_size = image_crop
    return wrapper


def get_layers(name):
    assert name == 'effnetb1_cutmixpatch_augmix_robust32_avge4e7_manylayers_324x288'
    return ['blocks', 'blocks.0', 'blocks.0.0', 'blocks.0.1',
            'blocks.1', 'blocks.1.0', 'blocks.1.1', 'blocks.1.2',
            'blocks.2', 'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
            'blocks.3', 'blocks.3.0', 'blocks.3.1', 'blocks.3.2', 'blocks.3.3',
            'blocks.4', 'blocks.4.0',
            'blocks.4.0.conv_pw', 'blocks.4.0.conv_dw', 'blocks.4.0.conv_pwl', 'blocks.4.1', 'blocks.4.1.conv_pw',
            'blocks.4.1.conv_dw', 'blocks.4.1.conv_pwl', 'blocks.4.2',
            'blocks.4.2.conv_pw', 'blocks.4.2.conv_dw', 'blocks.4.2.conv_pwl', 'blocks.4.3', 'blocks.4.3.conv_pw',
            'blocks.4.3.conv_dw', 'blocks.4.3.conv_pwl', 'blocks.5',
            'blocks.5.0', 'blocks.5.0.conv_pw', 'blocks.5.0.conv_dw', 'blocks.5.0.conv_pwl', 'blocks.5.1',
            'blocks.5.1.conv_pw', 'blocks.5.1.conv_dw', 'blocks.5.1.conv_pwl',
            'blocks.5.2', 'blocks.5.2.conv_pw', 'blocks.5.2.conv_dw', 'blocks.5.2.conv_pwl', 'blocks.5.3',
            'blocks.5.3.conv_pw', 'blocks.5.3.conv_dw', 'blocks.5.3.conv_pwl',
            'blocks.5.4', 'blocks.5.4.conv_pw', 'blocks.5.4.conv_dw', 'blocks.5.4.conv_pwl', 'blocks.6', 'blocks.6.0',
            'blocks.6.0.conv_pw', 'blocks.6.0.conv_dw',
            'blocks.6.0.conv_pwl', 'blocks.6.1', 'blocks.6.1.conv_pw', 'blocks.6.1.conv_dw', 'blocks.6.1.conv_pwl',
            'global_pool', 'global_pool.flatten', 'global_pool.pool']


def get_bibtex(model_identifier):
    return """@InProceedings{pmlr-v97-tan19a,
                title = 	 {{E}fficient{N}et: Rethinking Model Scaling for Convolutional Neural Networks},
                author =       {Tan, Mingxing and Le, Quoc},
                booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
                pages = 	 {6105--6114},
                year = 	 {2019},
                editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
                volume = 	 {97},
                series = 	 {Proceedings of Machine Learning Research},
                month = 	 {09--15 Jun},
                publisher =    {PMLR},
                pdf = 	 {http://proceedings.mlr.press/v97/tan19a/tan19a.pdf},
                url = 	 {https://proceedings.mlr.press/v97/tan19a.html},
                abstract = 	 {Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are given. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves stateof-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet (Huang et al., 2018). Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flower (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.}
                }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
