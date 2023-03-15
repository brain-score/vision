import pytest
from pathlib import Path

from candidate_models.base_models.cornet import cornet


@pytest.mark.memory_intense
def test_no_time_separation():
    model = cornet('CORnet-S', separate_time=False)
    stimulus_path = Path(__file__).parent / 'rgb.jpg'
    layers = ['IT.output-t0', 'IT.output-t1']
    activations = model([stimulus_path], layers=layers)
    assert set(activations['layer'].values) == set(layers)
