import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import torch
from brainscore_vision.model_helpers.s3 import load_weight_file
from timm.models import create_model
from brainscore_vision.model_helpers.activations.pytorch import load_images
import ssl
import numpy as np
from torchvision import transforms


ssl._create_default_https_context = ssl._create_unverified_context
INPUT_SIZE = 256
BATCH_SIZE = 64
LAYERS = ['blocks.1.blocks.1.4.norm2']


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.CenterCrop((image_size,image_size)),
        torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])


def load_preprocess_custom_model(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images

def get_model(name):
    assert name == 'custom_model_cv_18_dagger_408'
    model = create_model('crossvit_18_dagger_408', pretrained=False)
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="custom_model_cv_18_dagger_408/crossvit_18_dagger_408_adv_finetuned_epoch5.pt",
                                    version_id="pQVPFC_iiWpRRr7P54qxQfRzjNSn2uYB",
                                    sha1="c769518485e352d5a2e6f3e588d6208cbad71b69")
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_custom_model, image_size=224)
    activations_model = PytorchWrapper(identifier='custom_model_cv_18_dagger_408', model=model,
                                       preprocessing=preprocessing, batch_size=BATCH_SIZE)
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'custom_model_cv_18_dagger_408'
    return LAYERS


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
