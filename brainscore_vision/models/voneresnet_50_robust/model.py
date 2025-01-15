import functools
from vonenet import get_model as create_model
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models

# layers = {
#     'voneresnet-50-robust':
#         ['vone_block'] +
#         ['model.layer1.0.conv3', 'model.layer1.1.conv3', 'model.layer1.2.conv3'] +
#         ['model.layer2.0.downsample.0', 'model.layer2.1.conv3', 'model.layer2.2.conv3', 'model.layer2.3.conv3'] +
#         ['model.layer3.0.downsample.0', 'model.layer3.1.conv3', 'model.layer3.2.conv3', 'model.layer3.3.conv3',
#          'model.layer3.4.conv3', 'model.layer3.5.conv3'] +
#         ['model.layer4.0.downsample.0', 'model.layer4.1.conv3', 'model.layer4.2.conv3'] +
#         ['model.avgpool'],}


def voneresnet(model_name='resnet50_at'):
    model = create_model(model_name)
    model = model.module
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier='vone'+model_name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_model(name):
    assert name == "voneresnet-50-robust"
    return voneresnet()

def get_layers(name):
    assert name == "voneresnet-50-robust"

    layers = ['vone_block',
        'model.layer1.0', 'model.layer1.1', 'model.layer1.2',
        'model.layer2.0', 'model.layer2.1', 'model.layer2.2', 'model.layer2.3',
        'model.layer3.0', 'model.layer3.1', 'model.layer3.2', 'model.layer3.3',
        'model.layer3.4', 'model.layer3.5',
        'model.layer4.0', 'model.layer4.1', 'model.layer4.2',
        'model.avgpool']

    return layers

def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''' '''

if __name__ == '__main__':
    check_models.check_base_models(__name__)
