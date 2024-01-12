import functools

import torch
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import numpy as np
import timm
import torch.nn as nn
from albumentations import Compose, Normalize, Resize, CenterCrop
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from brainscore_vision.model_helpers import download_weights

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models
from collections import OrderedDict
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

image_resize = 247
image_crop = 234
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
"""
freeze_layers = ['blocks.0.0', 'blocks.0.1', 'blocks.1.0', 
                'blocks.1.1', 'blocks.1.2', 'blocks.2.0', 
                'blocks.2.1', 'blocks.2.2', 'blocks.3.0']
"""


def custom_image_preprocess(images, **kwargs):
    transforms_val = Compose(
        [
            Resize(image_resize, image_resize),
            CenterCrop(image_crop, image_crop),
            Normalize(
                mean=norm_mean,
                std=norm_std,
            ),
            ToTensorV2(),
        ]
    )

    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)

    return images


def load_preprocess_images_custom(
    image_filepaths, preprocess_images=custom_image_preprocess, **kwargs
):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if (
            "L" not in pil_image.mode.upper()
            and "A" not in pil_image.mode.upper()
            and "P" not in pil_image.mode.upper()
        ):  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


class EffNetBX(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.efnet_model = timm.create_model("resnet50", pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x


def get_model():
    model = EffNetBX()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    download_weights(
        bucket="brainscore-vision",
        folder_path="models/test_robust_sub",
        filename_version_sha=[
            (
                "resnet50_robust_cutmixfrontpatchres_e2e3.pth",
                "sQYiDIza7vdlY1HPv.kfT2.Iz_ofsAYW",
                "26c09e6ccac6d98bb8d384da35ed59da8f2ffa66",
            )
        ],
        save_directory=Path(__file__).parent,
    )
    state_dict = torch.load(
        dir_path + "/resnet50_robust_cutmixfrontpatchres_e2e3.pth",
        map_location=torch.device("cpu"),
    )["model"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # strip `module.` prefix
        if "efnet" in k and not "attacker" in k:
            name = k.replace("module.model.", "")
            new_state_dict[name] = v

    res = model.load_state_dict(new_state_dict)
    print(res)
    model = model.efnet_model

    filter_elems = set(["se", "act", "bn", "conv"])
    layer_list = [
        layer
        for layer, _ in model.named_modules()
        if not any(i in layer for i in filter_elems)
    ]
    print(layer_list)
    print(len(layer_list))

    for n, m in model.named_modules():
        if isinstance(
            m, nn.BatchNorm2d
        ):  # and any(x in n for x in ["conv_stem" ] + freeze_layers) or n =="bn1":
            print(f"Freeze {n, m}")
            m.eval()

    preprocessing = functools.partial(
        load_preprocess_images_custom,
        preprocess_images=custom_image_preprocess,
    )

    wrapper = PytorchWrapper(
        identifier=name, model=model, preprocessing=preprocessing, batch_size=8
    )

    wrapper.image_size = image_crop
    return wrapper


LAYERS = [
    "conv1",
    "layer2.0.downsample",
    "layer2.0.downsample.0",
    "layer2.0.downsample.1",
    "layer2.1",
    "layer2.2",
    "layer2.3",
    "layer3",
    "layer3.0",
    "layer3.0.downsample",
    "layer3.0.downsample.0",
    "layer3.0.downsample.1",
    "layer3.1",
    "layer3.2",
    "layer3.3",
    "layer3.4",
    "layer3.5",
    "layer4",
    "layer4.0",
    "layer4.0.downsample",
    "layer4.0.downsample.0",
    "layer4.0.downsample.1",
    "layer4.1",
    "layer4.2",
    "global_pool",
    "global_pool.flatten",
    "global_pool.pool",
]


BIBTEX = """@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
