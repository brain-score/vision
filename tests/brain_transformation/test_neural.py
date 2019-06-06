import functools

import numpy as np
import pytest

from brainscore.assemblies.public import load_assembly
from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment


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
    @pytest.mark.parametrize(['model_ctr', 'layers', 'expected_layer', 'assembly_identifier', 'region'],
                             [(pytorch_custom, ['linear', 'relu2'], 'relu2', 'dicarlo.Majaj2015.lowvar.IT', 'IT')])
    def test(self, model_ctr, layers, expected_layer, assembly_identifier, region):
        np.random.seed(0)
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=layers)
        assembly = load_assembly(assembly_identifier, average_repetition=False)
        brain_model.commit_region(region, assembly, assembly_stratification='category_name')

        brain_model.start_recording(region, [(70, 170)])
        predictions = brain_model.look_at(assembly.stimulus_set)
        assert set(predictions['region'].values) == {region}
        assert set(predictions['layer'].values) == {expected_layer}
