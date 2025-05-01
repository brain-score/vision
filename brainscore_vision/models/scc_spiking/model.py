from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from spikingjelly.activation_based import neuron


class SimpleSpikingModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSpikingModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
        self.spike1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0)
        conv_output_size = (224 - 3 + 1)  # 222
        linear_input_size = conv_output_size * conv_output_size * 2
        self.linear = torch.nn.Linear(linear_input_size, 1000)
        self.spike2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0)

    def forward(self, x):
        # Reset membrane potentials at the start of every forward pass
        for module in self.modules():
            if hasattr(module, 'reset'):
                module.reset()

        x = self.conv1(x)
        x = self.spike1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.spike2(x)
        return x


def get_model_list():
    return ['simple_spiking_model']


def get_model(name):
    assert name == 'simple_spiking_model'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = SimpleSpikingModel()
    activations_model = PytorchWrapper(identifier='simple_spiking_model', model=model, preprocessing=preprocessing)
    wrapper = ModelCommitment(identifier='simple_spiking_model',
                              activations_model=activations_model,
                              layers=['conv1', 'spike1', 'spike2'])
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'simple_spiking_model'
    return ['conv1', 'spike1', 'spike2']


def get_bibtex(model_identifier):
    return """
    """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
