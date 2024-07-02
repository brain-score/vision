# Created by David Coggan on 2024 06 26

import brainscore_vision
import numpy as np


def test_Coggan2024_fMRI_stimuli():
    stimulus_set = brainscore_vision.load_stimulus_set('Coggan2024_fMRI')
    assert len(stimulus_set) == 24
    assert len(set(stimulus_set['object_name'])) == 8
    assert len(set(stimulus_set['occlusion_condition'])) == 3


def test_Coggan2024_fMRI_dataset():
    assembly = brainscore_vision.load_dataset('Coggan2024_fMRI')
    np.testing.assert_array_equal(
        assembly.dims, ['presentation', 'presentation', 'neuroid'])
    assert len(set(assembly['stimulus_id'].values)) == 24
    assert assembly.shape[0] == 24
    assert assembly.shape[1] == 24
    assert assembly.shape[2] == 36
    assert assembly.stimulus_set is not None
    assert len(assembly.stimulus_set) == 24


