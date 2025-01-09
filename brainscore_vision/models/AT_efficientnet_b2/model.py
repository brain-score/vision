import functools
from efficientnet_pytorch import EfficientNet
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models


def get_model(name):
    assert name == 'AT_efficientnet-b2'
    model = EfficientNet.from_pretrained("efficientnet-b2", advprop=True)
    model.set_swish(memory_efficient=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    from types import MethodType
    def _output_layer(self):
        return self._model._fc

    wrapper._output_layer = MethodType(_output_layer, wrapper)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'AT_efficientnet-b2'
    return [
        '_blocks.0',
        '_blocks.1',
        '_blocks.2',
        '_blocks.3',
        '_blocks.4',
        '_blocks.5',
        '_blocks.6',
        '_blocks.7',
        '_blocks.8',
        '_blocks.9',
        '_blocks.10',
        '_blocks.11',
        '_blocks.12',
        '_blocks.13',
        '_blocks.14',
        '_blocks.15',
        '_blocks.16',
        '_blocks.17',
        '_blocks.18',
        '_blocks.19',
        '_blocks.20',
        '_blocks.21',
        '_blocks.22',
    ]

def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''

if __name__ == '__main__':
    check_models.check_base_models(__name__)
