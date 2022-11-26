import numpy as np
from numpy.random import RandomState

from brainscore_vision import BrainModel
from brainscore_vision import load_model
from brainscore_vision.utils import get_sample_stimuli_paths, get_sample_stimulus_set


def test_probabilities():
    model = load_model('pixels')
    stimuli = get_sample_stimulus_set()
    stimuli['label'] = RandomState(0).choice(['label1', 'label2', 'label3'], size=len(stimuli))
    model.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=stimuli)
    probabilities = model.look_at(stimuli)['behavior']
    assert len(probabilities['presentation']) == len(stimuli)
    assert len(probabilities['choice']) == 3
    expected_probabilities = ...  # TODO
    np.testing.assert_allclose(probabilities, expected_probabilities, atol=0.0001)


def test_neural():
    model = load_model('pixels')
    stimuli = get_sample_stimuli_paths()
    model.start_recording(recording_target=BrainModel.RecordingTarget.IT, time_bins=[(70, 170)])
    activations = model.look_at(stimuli)['neural']
    assert len(activations['presentation']) == len(stimuli)
    assert set(activations['stimulus_path'].values) == set(stimuli)
    expected_feature_size = 256 * 256 * 3
    assert len(activations['neuroid']) == expected_feature_size
