import functools

import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models


def get_model_list():
    return ['vgg16']


def get_model(name):
    assert name == 'vgg16'
    model = torchvision.models.vgg16(pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='vgg16', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'vgg16'
    return ['features.4', 'features.9', 'features.16', 'features.23', 'features.30',
            'classifier.1', 'classifier.4']


def get_bibtex(model_identifier):
    return """Blah Blah"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
