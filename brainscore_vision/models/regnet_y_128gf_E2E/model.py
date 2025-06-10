# Created by David Coggan on 2025 03 13
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == 'regnet_y_128gf_E2E'
    model = torchvision.models.regnet_y_128gf(weights='IMAGENET1K_SWAG_E2E_V1')
    preprocessing = functools.partial(load_preprocess_images, image_size=384)
    wrapper = PytorchWrapper(identifier='regnet_y_128gf_E2E', model=model,
                             preprocessing=preprocessing)
    wrapper.image_size = 384
    return wrapper

def get_layers(name):
    assert name == 'regnet_y_128gf_E2E'
    return ['trunk_output.block1', 'trunk_output.block2',
            'trunk_output.block3', 'trunk_output.block4', 'fc']


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)