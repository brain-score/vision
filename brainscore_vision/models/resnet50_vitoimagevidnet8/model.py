from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.s3 import load_weight_file
import functools
import torch
from .helpers.helpers import resnet50

"""
Template module for a base model submission to brain-score
"""



def get_model(name):
    assert name == 'resnet50-vitoimagevidnet8'
    model = resnet50(attn_pool=True)
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="resnet50-vitoimagevidnet8/resnet50_vito_linear_sdattn.pth",
                                    version_id="null",
                                    sha1="2103a1941017fdb5f58ac77e5587e7a73cd96bd8")
    r50_vito_sd = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(r50_vito_sd, strict=False)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='resnet50-vitoimagevidnet8', model=model, preprocessing=preprocessing)
    return wrapper


def get_layers(name):
    assert name == 'resnet50-vitoimagevidnet8'
    layers = ['conv1', 'layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
              'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1',
              'layer4.2', 'avgpool', 'avgpool2', 'fc']
    return layers


def get_bibtex(model_identifier):
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_brain_models(__name__)
