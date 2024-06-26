from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from model_helpers.activations.pytorch import load_preprocess_images
import ssl
import functools
import antialiased_cnns
from brainscore_vision.model_helpers.check_submission import check_models

ssl._create_default_https_context = ssl._create_unverified_context

def get_model(name):
    assert name == 'antialiased-rnext101_32x8d'
    model = antialiased_cnns.resnext101_32x8d(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='antialiased-rnext101_32x8d', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'antialiased-rnext101_32x8d'
    return ['maxpool', 'layer1.0', 'layer1.1', 'layer1.2',
            'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3'] + ['layer3.' + str(i) for i in range(23)] + ['layer4.0',
                                                                                                         'layer4.1',
                                                                                                         'layer4.2',
                                                                                                         'avgpool',
                                                                                                         'fc']

def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)