import functools

from model_tools.activations.pytorch import load_preprocess_images, PytorchWrapper
from model_tools.brain_transformation import ModelCommitment
from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""

# model = AACN_ResNet.resnet50(num_classes=8, attention=[False, True, True, True], num_heads=4, k=2, v=0.25, image_size=224)
import torchvision.models as models
model = models.resnet50()

preprocessing = functools.partial(load_preprocess_images, image_size=224)
activations_model = PytorchWrapper(identifier='resnet50_test', model=model,
                                   preprocessing=preprocessing)
# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='resnet50_test', activations_model=activations_model,
                        # specify layers to consider
                        layers=['conv1'] +
                               ['layer1.0.conv3', 'layer1.1.conv3', 'layer1.2.conv3'] +
                               ['layer2.0.downsample.0', 'layer2.1.conv3', 'layer2.2.conv3', 'layer2.3.conv3'] +
                               ['layer3.0.downsample.0', 'layer3.1.conv3', 'layer3.2.conv3', 'layer3.3.conv3',
                                'layer3.4.conv3', 'layer3.5.conv3'] +
                               ['layer4.0.downsample.0', 'layer4.1.conv3', 'layer4.2.conv3', 'avgpool'])


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['resnet50_test']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """

    assert name == 'resnet50_test'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return ['conv1', 'layer1.0.conv3', 'layer1.1.conv3', 'layer1.2.conv3', 'layer2.0.downsample.0', 'layer2.1.conv3',
            'layer2.2.conv3', 'layer2.3.conv3', 'layer3.0.downsample.0', 'layer3.1.conv3', 'layer3.2.conv3',
            'layer3.3.conv3', 'layer3.4.conv3', 'layer3.5.conv3', 'layer4.0.downsample.0', 'layer4.1.conv3',
            'layer4.2.conv3', 'avgpool']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
