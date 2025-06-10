from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import sew_resnet,spiking_resnet
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


def get_model_list():
    return ['sewresnet101_1']

def get_model(name):
    assert name == 'sewresnet101_1'
    model = sew_resnet.sew_resnet101(pretrained=False, progress=True, cnf='ADD',spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='sewresnet101_1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
  assert name == 'sewresnet101_1'
  return ['conv1','layer1', 'layer2', 'layer3', 'layer4', 'fc']


def get_bibtex(model_identifier):
  return """"""


if __name__ == '__main__':
  check_models.check_base_models(__name__)