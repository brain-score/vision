# Created by David Coggan on 2024 06 26

import pytest
import numpy as np
import brainscore_vision

@pytest.mark.private_access
def test_Coggan2024_behavior_stimuli():
    stimulus_set = brainscore_vision.load_stimulus_set('Coggan2024_behavior')
    assert len(stimulus_set) == 22560
    assert len(set(stimulus_set['object_class'])) == 8
    assert len(set(stimulus_set['occluder_type'])) == 10
    assert len(set(stimulus_set['occluder_color'])) == 3
    assert len(set(stimulus_set['visibility'])) == 6

@pytest.mark.private_access
def test_Coggan2024_behavior_stimuli_fitting():
    stimulus_set = brainscore_vision.load_stimulus_set(
        'Coggan2024_behavior_fitting')
    assert len(stimulus_set) == 2048

@pytest.mark.private_access
def test_Coggan2024_behavior_dataset():
    assembly = brainscore_vision.load_dataset('Coggan2024_behavior')
    np.testing.assert_array_equal(
        assembly.dims, ['presentation'])
    assert len(set(assembly['stimulus_id'].values)) == 22560
    assert assembly.shape[0] == 22560
    assert assembly.stimulus_set is not None
    assert len(assembly.stimulus_set) == 22560


