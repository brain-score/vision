import pytest
import numpy as np
from brainscore_vision import load_dataset, load_stimulus_set

class TestAssemblyProperties:
    @pytest.mark.parametrize('assembly', ['Gifford2022_train', 'Gifford2022_test'])
    def test_assembly_existence(self, assembly):
        assert load_dataset(assembly) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 1654, 10, 4),
        ('test', 200, 1, 80),
    ])
    def test_assembly_shape(self, split, n_categories, n_examples, n_repetitions):
        assembly = load_dataset(f'Gifford2022_{split}')
        # assembly shape: 
        # (n_categories * n_examples * n_repetitions, n_electrodes * n_subjects, n_timepoints)
        assert assembly.shape == (n_categories * n_examples * n_repetitions, 17 * 10, 100), f"Got shape {assembly.shape}"
        
        # Check all expected coords exist
        expected_presentation_coords = ['stimulus_id', 'stimulus_label', 'object_name', 'stimulus_label_idx', 'repetition']
        expected_neuroid_coords = ['subject', 'neuroid_id', 'channel']
        expected_time_bin_coords = ['time_bin_start', 'time_bin_end']
        
        existing_coords = set()
        for dim in assembly.dims:
            index = assembly.coords[dim].to_index()
            if hasattr(index, "names"):
                existing_coords.update(index.names)

        for coord in expected_presentation_coords:
            assert coord in existing_coords, f"Expected coord '{coord}' not found in assembly"
            assert assembly.coords[coord].dims == ('presentation',), f"Coord '{coord}' has unexpected dims: {assembly.coords[coord].dims}"
        
        for coord in expected_neuroid_coords:
            assert coord in existing_coords, f"Expected coord '{coord}' not found in assembly"
            assert assembly.coords[coord].dims == ('neuroid',), f"Coord '{coord}' has unexpected dims: {assembly.coords[coord].dims}"
        
        for coord in expected_time_bin_coords:
            assert coord in existing_coords, f"Expected coord '{coord}' not found in assembly"
            assert assembly.coords[coord].dims == ('time_bin',), f"Coord '{coord}' has unexpected dims: {assembly.coords[coord].dims}"

        #check that time bins are correct (-200ms to 800ms in 10ms steps)
        assert list(assembly.coords['time_bin'].values) == [(np.round(bin, 2), np.round(bin + 0.01, 2)) for bin in np.arange(-0.2, 0.8, 0.01)],\
            "Time bins do not match expected values."

@pytest.mark.private_access
class TestStimulusSetProperties:
    @pytest.mark.parametrize('stim_set', ['Gifford2022_stim_train', 'Gifford2022_stim_test'])
    def test_stimulus_set_existence(self, stim_set):
        assert load_stimulus_set(stim_set) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 1654, 10, 4),
        ('test', 200, 1, 80),
    ])
    def test_stimulus_contents(self, split, n_categories, n_examples, n_repetitions):
        stimulus_set = load_stimulus_set(f'Gifford2022_stim_{split}')
        assembly = load_dataset(f'Gifford2022_{split}')
        assert len(stimulus_set) == n_categories * n_examples, f"Got length {len(stimulus_set)}"
        assert len(stimulus_set['label'].unique()) == n_categories, f"Number of unique labels in stimulus set is not {n_categories}."
        assert np.array_equal(
            np.repeat(stimulus_set['stimulus_id'], n_repetitions),
            assembly['stimulus_id'].values
        ), f"Stimulus IDs in Gifford2022_stim_{split} stimulus set and assembly do not match."