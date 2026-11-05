import numpy as np
import torch

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper


def test_device_argument_overrides_the_automatic_choice():
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
    wrapper = PytorchWrapper(network, preprocessing=lambda paths: np.zeros((len(paths), 3, 2, 2)),
                             identifier='device-test', device='cpu')
    assert wrapper._device == torch.device('cpu')
    assert next(wrapper._model.parameters()).device == torch.device('cpu')
