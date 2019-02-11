import functools

import numpy as np
import pytest

from model_tools.activations import PytorchWrapper
from model_tools.multilayer_mapping import ModelCommitment
from model_tools.multilayer_mapping.data import it_translation_data


def pytorch_custom():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.relu2(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=MyModel(), preprocessing=preprocessing)


class TestLayerSelection:
    @pytest.mark.parametrize(['model_ctr', 'layers', 'expected_layer', 'data_ctr', 'region'],
                             [(pytorch_custom, ['linear', 'relu2'], 'relu2', it_translation_data, 'IT')])
    def test(self, model_ctr, layers, expected_layer, data_ctr, region):
        np.random.seed(0)
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, base_model=activations_model,
                                      layers=layers)
        assembly = data_ctr()
        brain_model.commit_region(region, assembly)

        brain_model.start_recording(region)
        predictions = brain_model.look_at(assembly.stimulus_set)
        assert set(predictions['region'].values) == {region}
        assert set(predictions['layer'].values) == {expected_layer}
