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
    #models = ['efficientnet-b0', 'efficientnet-b2', 'efficientnet-b4', 'efficientnet-b6']
    models = ['efficientnet-b7', 'efficientnet-b8']

    models = ['AT_'+model for model in models]
    return models

def get_model(name):
    model = name
    AT = False
    if 'AT_' in name:
        AT = True
        model = model.split('AT_')[-1]

    model = EfficientNet.from_pretrained(model, advprop=AT)
    model.set_swish(memory_efficient=False)

    if AT:
        preprocessing = functools.partial(load_preprocess_images, image_size=224, normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    else:
        preprocessing = functools.partial(load_preprocess_images, image_size=224)

    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    from types import MethodType
    def _output_layer(self):
        return self._model._fc

    wrapper._output_layer = MethodType(_output_layer, wrapper)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    name = name.split('AT_')[-1]
    assert name in lmap
    return lmap[name]

lmap = {
    'efficientnet-b0' : [f'_blocks.{i}' for i in range(16)],
    'efficientnet-b2' : [f'_blocks.{i}' for i in range(23)],
    'efficientnet-b4' : [f'_blocks.{i}' for i in range(32)],
    'efficientnet-b6' : [f'_blocks.{i}' for i in range(45)],
    'efficientnet-b7' : [f'_blocks.{i}' for i in range(55)],
    'efficientnet-b8' : [f'_blocks.{i}' for i in range(61)]
}

if __name__ == '__main__':
    check_models.check_base_models(__name__)
