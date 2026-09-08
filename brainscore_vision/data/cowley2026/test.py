import numpy as np
import pytest
from brainscore_vision import load_dataset, load_stimulus_set

SESSIONS = {'190923': 1200, '201025': 1200, '210225': 1200, '211022': 1600}


@pytest.mark.private_access
class TestStimulusSet:
    @pytest.mark.parametrize('session, num_stimuli', SESSIONS.items())
    def test_stimulus_set(self, session, num_stimuli):
        stimulus_set = load_stimulus_set(f'Cowley2026.{session}')
        assert stimulus_set is not None
        assert stimulus_set.identifier == f'Cowley2026.{session}'
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == num_stimuli


@pytest.mark.private_access
class TestAssembly:
    @pytest.mark.parametrize('session', list(SESSIONS))
    def test_assembly(self, session):
        assembly = load_dataset(f'Cowley2026.{session}')
        assert assembly is not None
        assert assembly.identifier == f'Cowley2026.{session}'
        assert 'stimulus_id' in assembly.indexes['presentation'].names
        assert set(np.unique(assembly['region'].values)) == {'V4'}
        assembly_stimuli = set(assembly['stimulus_id'].values)
        assert assembly_stimuli.issubset(set(assembly.stimulus_set['stimulus_id'].values))
