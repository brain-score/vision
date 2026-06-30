import numpy as np
import pytest
from brainscore_vision import load_dataset, load_stimulus_set

@pytest.mark.private_access
class TestStimulusSet:
    def test_stimulus_set_exists(self):
        """Test that stimulus set loads correctly"""
        stimulus_set = load_stimulus_set('Cowley2026.190923')
        assert stimulus_set is not None
        assert stimulus_set.identifier == 'Cowley2026.190923'

        # stimulus_set = load_stimulus_set('Cowley2026.201025')
        # assert stimulus_set is not None
        # assert stimulus_set.identifier == 'Cowley2026.201025'

        # stimulus_set = load_stimulus_set('Cowley2026.210225')
        # assert stimulus_set is not None
        # assert stimulus_set.identifier == 'Cowley2026.210225'

        # stimulus_set = load_stimulus_set('Cowley2026.211022')
        # assert stimulus_set is not None
        # assert stimulus_set.identifier == 'Cowley2026.211022'

        

    def test_stimulus_set_counts(self):
        """Test expected number of stimuli"""
        stimulus_set = load_stimulus_set('Cowley2026.190923')
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1200

        # stimulus_set = load_stimulus_set('Cowley2026.201025')
        # assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1200

        # stimulus_set = load_stimulus_set('Cowley2026.210225')
        # assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1200

        # stimulus_set = load_stimulus_set('Cowley2026.211022')
        # assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1600




@pytest.mark.private_access
class TestAssembly:
    def test_assembly_exists(self):
        """Test that assembly loads correctly"""
        assembly = load_dataset('Cowley2026.190923')
        assert assembly is not None
        assert assembly.identifier == 'Cowley2026.190923'

        # assembly = load_dataset('Cowley2026.201025')
        # assert assembly is not None
        # assert assembly.identifier == 'Cowley2026.201025'

        # assembly = load_dataset('Cowley2026.210225')
        # assert assembly is not None
        # assert assembly.identifier == 'Cowley2026.210225'

        # assembly = load_dataset('Cowley2026.211022')
        # assert assembly is not None
        # assert assembly.identifier == 'Cowley2026.211022'



    def test_assembly_structure(self):
        """Test assembly has required dimensions and coordinates"""
        assembly = load_dataset('Cowley2026.190923')
        assert 'presentation' in assembly.dims
        assert 'stimulus_id' in assembly.indexes['presentation'].names

        # assembly = load_dataset('Cowley2026.201025')
        # assert 'presentation' in assembly.dims
        # assert 'stimulus_id' in assembly.indexes['presentation'].names

        # assembly = load_dataset('Cowley2026.210225')
        # assert 'presentation' in assembly.dims
        # assert 'stimulus_id' in assembly.indexes['presentation'].names

        # assembly = load_dataset('Cowley2026.211022')
        # assert 'presentation' in assembly.dims
        # assert 'stimulus_id' in assembly.indexes['presentation'].names


    def test_assembly_alignment(self):
        """Test stimulus set and assembly are properly linked"""
        assembly = load_dataset('Cowley2026.190923')
        stimulus_set = assembly.stimulus_set

        # All assembly stimulus IDs should exist in stimulus set
        assembly_stimuli = set(assembly['stimulus_id'].values)
        stimulus_set_stimuli = set(stimulus_set['stimulus_id'].values)
        assert assembly_stimuli.issubset(stimulus_set_stimuli)

        # assembly = load_dataset('Cowley2026.201025')
        # stimulus_set = assembly.stimulus_set
        # assembly_stimuli = set(assembly['stimulus_id'].values)
        # stimulus_set_stimuli = set(stimulus_set['stimulus_id'].values)
        # assert assembly_stimuli.issubset(stimulus_set_stimuli)

        # assembly = load_dataset('Cowley2026.210225')
        # stimulus_set = assembly.stimulus_set
        # assembly_stimuli = set(assembly['stimulus_id'].values)
        # stimulus_set_stimuli = set(stimulus_set['stimulus_id'].values)
        # assert assembly_stimuli.issubset(stimulus_set_stimuli)

        # assembly = load_dataset('Cowley2026.211022')
        # stimulus_set = assembly.stimulus_set
        # assembly_stimuli = set(assembly['stimulus_id'].values)
        # stimulus_set_stimuli = set(stimulus_set['stimulus_id'].values)
        # assert assembly_stimuli.issubset(stimulus_set_stimuli)




