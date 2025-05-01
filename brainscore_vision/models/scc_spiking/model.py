from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

# Import from SpikingJelly
from spikingjelly.clock_driven import neuron, encoding, functional, layer

class MySpikingModel(torch.nn.Module):
    def __init__(self):
        super(MySpikingModel, self).__init__()
        # Example: Poisson encoding and a simple spiking layer
        self.encoder = encoding.PoissonEncoder()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=3)
        self.sn = neuron.IFNode()  # Integrate-and-fire neuron
        self.fc = torch.nn.Linear((224 - 2)**2 * 2, 1000)

    def forward(self, x):
        x = x / 255.0  # normalize if needed
        # Repeat input across time steps (T, N, C, H, W)
        T = 10  # number of time steps
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        x = self.encoder(x)
        spk_out = []

        for t in range(T):
            out = self.conv(x[t])
            out = self.sn(out)
            spk_out.append(out)

        out = sum(spk_out) / T  # average across time
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_list():
    return ['simple_spiking_model']

def get_model(name):
    assert name == 'simple_spiking_model'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = MySpikingModel()
    wrapper = PytorchWrapper(identifier='simple_spiking_model', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'simple_spiking_model'
    return ['conv', 'sn']  # only layers Brain-Score will evaluate

def get_bibtex(model_identifier):
    return """@article{spikingjelly,
    title={SpikingJelly: A Reproducible and Extensible Research Framework for Spiking Neural Network},
    author={Fang, Wei et al.},
    journal={arXiv preprint arXiv:2109.13264},
    year={2021}
    }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)





