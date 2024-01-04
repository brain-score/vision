import pytest

from brainscore import get_stimulus_set
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers


@pytest.mark.memory_intense
class TestActivations:
    @pytest.mark.parametrize('model_identifier', [
        'alexnet',
        'resnet-101_v2',
        'mobilenet_v2_1.0_224',
    ])
    def test_model(self, model_identifier):
        layers = model_layers[model_identifier]
        activations_model = base_model_pool[model_identifier]
        stimulus_set = get_stimulus_set('dicarlo.hvm')
        stimulus_set = stimulus_set[:100]
        stimulus_set.name = 'dicarlo.hvm-min'
        activations = activations_model(stimulus_set, layers=layers)
        assert activations is not None
        assert hasattr(activations, 'layer')
