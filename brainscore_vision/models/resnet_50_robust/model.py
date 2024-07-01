import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
import torch
from importlib import import_module
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import ssl
from brainscore_vision.model_helpers.s3 import load_weight_file


ssl._create_default_https_context = ssl._create_unverified_context

def get_model(name):
    assert name == 'resnet-50-robust'
    module = import_module(f'torchvision.models')
    model_ctr = getattr(module, 'resnet50')
    model = model_ctr()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="resnet-50-robust/ImageNet.pt",
                                    version_id=".shHB0L_L9L3Mtco0Kf4EBP3Xj9nLKnC",
                                    sha1="cc6e4441abc8ad6d2f4da5db84836e544bfb53fd")
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    # process weights -- remove the attacker and prepocessing weights
    weights = checkpoint['model']
    weights = {k[len('module.model.'):]: v for k, v in weights.items() if 'attacker' not in k}
    weights = {k: weights[k] for k in list(weights.keys())[2:]}
    model.load_state_dict(weights)
    # wrap model with pytorch wrapper
    wrapper = PytorchWrapper(identifier='resnet50', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'resnet-50-robust'
    layers = (
            ['conv1'] +
            ['layer1.0.conv3', 'layer1.1.conv3', 'layer1.2.conv3'] +
            ['layer2.0.downsample.0', 'layer2.1.conv3', 'layer2.2.conv3', 'layer2.3.conv3'] +
            ['layer3.0.downsample.0', 'layer3.1.conv3', 'layer3.2.conv3', 'layer3.3.conv3',
             'layer3.4.conv3', 'layer3.5.conv3'] +
            ['layer4.0.downsample.0', 'layer4.1.conv3', 'layer4.2.conv3'] +
            ['avgpool']
    )
    return layers


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
