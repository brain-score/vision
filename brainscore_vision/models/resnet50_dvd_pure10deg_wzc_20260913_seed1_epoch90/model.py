"""ResNet-50 with frozen weights and the supplied Brain-Score preprocessing."""
import functools
import hashlib

import numpy as np
from PIL import Image
import torch
import torchvision.models
import torchvision.transforms as transforms

from brainscore_core.supported_data_standards.brainio.s3 import load_file
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

MODEL_NAME = 'resnet50_dvd_pure10deg_wzc_20260920_seed1_epoch90'
WEIGHT_BUCKET = 'brainscore-storage'
WEIGHT_FOLDER = 'brainscore-vision/models/user_784/'
WEIGHT_FILENAME = 'resnet50_dvd_pure10deg_wzc_20260920_seed1_epoch90.pth'
WEIGHT_VERSION_ID = 'NTzg0ZrR4u4Eu2eOob7PD6u3yu1EyVAV'
WEIGHT_SHA256 = '9c6c7ba7f7404e5c21815efc3f20cc2b92acb685a31709b1580374eff3144ccb'
device = torch.device('cpu')


def get_model_list():
    return [MODEL_NAME]


def get_model(name):
    assert name == MODEL_NAME
    if not WEIGHT_VERSION_ID:
        raise ValueError('Preview only: fill version_id in cloud_files.json, then run the build command.')
    # The uploaded file is an unchanged copy of this seed/step checkpoint.
    file_path = load_file(bucket=WEIGHT_BUCKET, folder_name=WEIGHT_FOLDER,
                          relative_path=WEIGHT_FILENAME, version_id=WEIGHT_VERSION_ID)
    checksum = hashlib.sha256()
    with open(str(file_path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    if checksum.hexdigest() != WEIGHT_SHA256:
        raise ValueError('The cloud weight file does not match the registered seed/epoch checkpoint.')
    checkpoint = torch.load(str(file_path), map_location='cpu', weights_only=True)
    # Only remove an actual DataParallel prefix; retain ordinary module names.
    state = {key[7:] if key.startswith('module.') else key: value
             for key, value in checkpoint['state_dict'].items()}
    if len(state) != len(checkpoint['state_dict']):
        raise ValueError('State-dict names collide after removing the module prefix.')
    model = torchvision.models.resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images_change, image_size=224)
    wrapper = PytorchWrapper(identifier=MODEL_NAME, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == MODEL_NAME
    # Preserve the supplied Brain-Score block outputs, including avgpool.
    return (['conv1'] + [f'layer1.{i}' for i in range(3)] +
            [f'layer2.{i}' for i in range(4)] +
            [f'layer3.{i}' for i in range(6)] +
            [f'layer4.{i}' for i in range(3)] + ['avgpool'])


def get_bibtex(model_identifier):
    return ''


def load_preprocess_images_change(image_filepaths, image_size, **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    return preprocess_images_change(images, image_size=image_size, **kwargs)


def load_image(image_filepath):
    # Retain the image-mode handling from the user's submitted model.
    with Image.open(image_filepath) as pil_image:
        if ('L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper()
                and 'P' not in pil_image.mode.upper()):
            return pil_image.copy()
        rgb_image = Image.new('RGB', pil_image.size)
        rgb_image.paste(pil_image)
        return rgb_image


def preprocess_images_change(images, image_size, **kwargs):
    # Preserve direct square resize and ToTensor; no normalization or developmental transform.
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        lambda img: img.unsqueeze(0),
    ])
    return np.concatenate([preprocess(image) for image in images])


if __name__ == '__main__':
    from brainscore_vision.model_helpers.check_submission import check_models
    check_models.check_base_models(__name__)
