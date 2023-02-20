import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

def load_preprocess_images_custom(image_filepaths, image_size=0, transform_timm=None, **kwargs):
    images = load_images(image_filepaths)
    if transform_timm is not None:
      images = [transform_timm(image) for image in images]
      images = np.stack(images)
    return images

def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]

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
    return ['effnetb0']


def get_model(name):
    assert name == 'effnetb0'
    model_tf_efficientnet_b0_ns= timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    config = resolve_data_config({}, model=model_tf_efficientnet_b0_ns)
    transform_timm = create_transform(**config)

    preprocessing = functools.partial(load_preprocess_images_custom, transform_timm=transform_timm)

    wrapper = PytorchWrapper(identifier='my-model', model=model_tf_efficientnet_b0_ns, preprocessing=preprocessing, batch_size=8)

    wrapper.image_size = config["input_size"][1]
    return wrapper


def get_layers(name):
    assert name == 'effnetb0'
    return ["blocks.0.0","blocks.1.1","blocks.2.0","blocks.2.1", "blocks.3.0", "blocks.3.1", "blocks.3.2", 
         "blocks.4.0","blocks.4.1","blocks.4.2", "blocks.5.0", "blocks.5.1", "blocks.5.2", "blocks.5.3", "blocks.6.0",
         "global_pool", "global_pool.flatten", "global_pool.pool"]


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
