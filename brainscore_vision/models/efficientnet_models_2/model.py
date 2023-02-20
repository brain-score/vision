import functools

from efficientnet_pytorch import EfficientNet
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
    models = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b4', 'efficientnet-b6']
    #models = ['efficientnet-b0']
    return models

def get_model(name):
    models = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b4', 'efficientnet-b6']
    assert name in models
    model = EfficientNet.from_pretrained(name)
    model.set_swish(memory_efficient=False)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    from types import MethodType
    def _output_layer(self):
        return self._model._fc

    wrapper._output_layer = MethodType(_output_layer, wrapper)
    import pdb; pdb.set_trace()
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    models = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b4', 'efficientnet-b6']
    assert name in models
    return lmap[name]

lmap = {
    'efficientnet-b0' : [
        '_blocks.0',
        '_blocks.1',
        '_blocks.2',
        '_blocks.3',
        '_blocks.4',
        '_blocks.5',
        '_blocks.6',
        '_blocks.7',
        '_blocks.7',
        '_blocks.9',
        '_blocks.10',
        '_blocks.11',
        '_blocks.12',
        '_blocks.13',
        '_blocks.14',
        '_blocks.15',
    ],
    'efficientnet-b2' : [
        '_blocks.0',
        '_blocks.1',
        '_blocks.2',
        '_blocks.3',
        '_blocks.4',
        '_blocks.5',
        '_blocks.6',
        '_blocks.7',
        '_blocks.7',
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
    ],
    'efficientnet-b4' : [
        '_blocks.0',
        '_blocks.1',
        '_blocks.2',
        '_blocks.3',
        '_blocks.4',
        '_blocks.5',
        '_blocks.6',
        '_blocks.7',
        '_blocks.7',
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
        '_blocks.23',
        '_blocks.24',
        '_blocks.25',
        '_blocks.26',
        '_blocks.27',
        '_blocks.28',
        '_blocks.29',
        '_blocks.30',
        '_blocks.31',
    ],
    'efficientnet-b6' : [
        '_blocks.0',
        '_blocks.1',
        '_blocks.2',
        '_blocks.3',
        '_blocks.4',
        '_blocks.5',
        '_blocks.6',
        '_blocks.7',
        '_blocks.7',
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
        '_blocks.23',
        '_blocks.24',
        '_blocks.25',
        '_blocks.26',
        '_blocks.27',
        '_blocks.28',
        '_blocks.29',
        '_blocks.30',
        '_blocks.31',
        '_blocks.32',
        '_blocks.33',
        '_blocks.34',
        '_blocks.35',
        '_blocks.36',
        '_blocks.37',
        '_blocks.38',
        '_blocks.39',
        '_blocks.40',
        '_blocks.41',
        '_blocks.42',
        '_blocks.43',
        '_blocks.44',
    ]
}

if __name__ == '__main__':
    check_models.check_base_models(__name__)
