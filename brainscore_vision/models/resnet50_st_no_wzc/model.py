from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.

# from brainscore_vision.model_helpers.s3 import load_file
# from brainscore_core.supported_data_standards.brainio.s3 import load_file
device = torch.device("cpu")

def get_model_list():
    return ['resnet50_st_no_wzc']

def get_model(name):
    assert name == 'resnet50_st_no_wzc'

    from brainscore_core.supported_data_standards.brainio.s3 import load_file
    file_path = load_file(bucket="brainscore-storage", folder_name="brainscore-vision/models/user_755/",
                          relative_path="st_no_checkpoint_89.pth",
                          version_id="lAS_bxHm3Fa4WjafmUBBhZ_bvj5jVnFK",
                          ),

    checkpoint = torch.load(str(file_path[0]), map_location=lambda storage, loc: storage)
    new_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        new_key = key.replace('module.', '', 1)
        new_state_dict[new_key] = value

    model = torchvision.models.__dict__["resnet50"]()
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    # model = torchvision.models.resnet50(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images_change, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet50_st_no_wzc', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

# def get_layers(name):
#     assert name == 'resnet50_st_wzc'
    #return ['conv1','layer1', 'layer2', 'layer3', 'layer4', 'fc']
def get_layers(name):
    assert name == 'resnet50_st_no_wzc'
    layer_names = (['conv1'] + [f'layer1.{i}' for i in range(3)] +
                   [f'layer2.{i}' for i in range(4)] +
                   [f'layer3.{i}' for i in range(6)] +
                   [f'layer4.{i}' for i in range(3)] + ['avgpool'])
    # layer_names = (['conv1'] + [f'layer1.{0}'] +
    #                [f'layer2.{0}'] +
    #                [f'layer3.{0}'] +
    #                [f'layer4.{0}'] + [f'layer4.{2}'] + ['avgpool'])
    return layer_names


def get_bibtex(model_identifier):
    return """"""

#--------------------------------------------------------------------


def load_preprocess_images_change(image_filepaths, image_size, **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths] #load_images(image_filepaths)
    images = preprocess_images_change(images, image_size=image_size, **kwargs)
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

def preprocess_images_change(images, image_size, **kwargs):
    preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            lambda img: img.unsqueeze(0)
        ])
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


if __name__ == '__main__':
    check_models.check_base_models(__name__)
