from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import torch
from .helpers.resnet import resnet50
from brainscore_vision.model_helpers.check_submission import check_models
from model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.s3 import load_weight_file

ssl._create_default_https_context = ssl._create_unverified_context


LAYERS = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0',
              'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'avgpool',
              'fc']

model = resnet50(attn_pool=True)
def get_model_list():
    return ['resnet50-VITO-8deg-cc']


def get_model(name):
    assert name == 'resnet50-VITO-8deg-cc'
    #model = resnet50(attn_pool=True)
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="resnet50-VITO-8deg-cc/resnet50_vito_linear_sdattn.pth",
                                    version_id="Ii7wJ1Invv0VgPsSSIG26NZx.S8r5jkb",
                                    sha1="2103a1941017fdb5f58ac77e5587e7a73cd96bd8")
    r50_vito_sd = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(r50_vito_sd, strict=False)
    model.eval()
    for name in model.named_modules():
         print(name)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet50-VITO-8deg-cc', model=model, preprocessing=preprocessing)
    vito_brain_model = ModelCommitment('resnet50-VITO-8deg-cc', activations_model=wrapper, layers=LAYERS)
    return wrapper  # original code returns the vito_brain_model, not wrapper, but this throws error, so we use wrapper


def get_layers(name):
    assert name == 'resnet50-VITO-8deg-cc'
    layers = [layer for layer,_ in model.named_modules()]
    #return LAYERSi
    print(layers)
    return layers[2:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
