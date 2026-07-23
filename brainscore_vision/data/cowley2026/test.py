import numpy as np
import pytest
from brainscore_vision import load_dataset, load_stimulus_set


@pytest.mark.private_access
class TestStimulusSet:
    def test_stimulus_set_exists(self):
        stimulus_set = load_stimulus_set('Cowley2026.190923')
        assert stimulus_set is not None
        assert stimulus_set.identifier == 'Cowley2026.190923'

    def test_stimulus_set_counts(self):
        stimulus_set = load_stimulus_set('Cowley2026.190923')
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1200


@pytest.mark.private_access
class TestAssembly:
    def test_assembly_exists(self):
        assembly = load_dataset('Cowley2026.190923')
        assert assembly is not None
        assert assembly.identifier == 'Cowley2026.190923'

    def test_assembly_structure(self):
        assembly = load_dataset('Cowley2026.190923')
        assert 'presentation' in assembly.dims
        assert 'stimulus_id' in assembly.indexes['presentation'].names
        assert set(np.unique(assembly['region'].values)) == {'V4'}

    def test_assembly_alignment(self):
        assembly = load_dataset('Cowley2026.190923')
        assembly_stimuli = set(assembly['stimulus_id'].values)
        stimulus_set_stimuli = set(assembly.stimulus_set['stimulus_id'].values)
        assert assembly_stimuli.issubset(stimulus_set_stimuli)
