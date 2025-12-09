import pytest
import numpy as np
from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers.neural_common import filter_reliable_neuroids
TOTAL_NEUROIDS = 2048
RELIABILITY_THRESHOLD = 0.3
EXPECTED_RELIAB_NEUROIDS = {
    'monkeyF': {'V1': 332, 'V4': 117, 'IT': 157},
    'monkeyN': {'V1': 410, 'V4': 198, 'IT': 141}
}
TOTAL_RELIAB_NEUROIDS = sum(
    sum(regions.values()) for regions in EXPECTED_RELIAB_NEUROIDS.values()
)

class TestAssemblyProperties:
    @pytest.mark.parametrize('assembly', ['Papale2025_train', 'Papale2025_test'])
    def test_assembly_existence(self, assembly):
        assert load_dataset(assembly) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 1854, 12, 1),
        ('test', 100, 1, 30),
    ])
    def test_assembly_shape(self, split, n_categories, n_examples, n_repetitions):
        assembly = load_dataset(f'Papale2025_{split}')
        # assembly shape:
        # (n_categories * n_examples * n_repetitions, n_neuroids, 1)
        assert assembly.shape == (n_categories * n_examples * n_repetitions, TOTAL_NEUROIDS, 1), f"Got shape {assembly.shape}"
        assert set(assembly['subject'].values) == set(EXPECTED_RELIAB_NEUROIDS.keys()), "unexpected subjects in assembly"
        assert set(assembly['region'].values) == set(EXPECTED_RELIAB_NEUROIDS['monkeyF'].keys()), "unexpected regions in assembly"
        
        reliab_assembly = filter_reliable_neuroids(assembly, RELIABILITY_THRESHOLD, 'reliability')
        for subject in EXPECTED_RELIAB_NEUROIDS.keys():
            for region in EXPECTED_RELIAB_NEUROIDS[subject].keys():
                assert len(reliab_assembly.sel(subject=subject).sel(region=region)['neuroid_id']) == EXPECTED_RELIAB_NEUROIDS[subject][region], \
                    f"Number of neuroids for subject {subject} in region {region} does not match expected."
        
        # Check all expected coords exist
        expected_presentation_coords = ['stimulus_id', 'stimulus_label', 'object_name', 'stimulus_label_idx', 'repetition']
        expected_neuroid_coords = ['region', 'subject', 'neuroid_id', 'SNR', 'SNR_max', 'reliability', 'oracle']
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

@pytest.mark.private_access
class TestStimulusSetProperties:

    @pytest.mark.parametrize('stim_set', ['Papale2025_stim_train', 'Papale2025_stim_test'])
    def test_stimulus_set_existence(self, stim_set):
        assert load_stimulus_set(stim_set) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 1854, 12, 1),
        ('test', 100, 1, 30),
    ])
    def test_stimulus_contents(self, split, n_categories, n_examples, n_repetitions):
        stimulus_set = load_stimulus_set(f'Papale2025_stim_{split}')
        assembly = load_dataset(f'Papale2025_{split}')
        assert len(stimulus_set) == n_categories * n_examples, f"Got length {len(stimulus_set)}"
        assert len(stimulus_set['label'].unique()) == n_categories, f"Number of unique labels in stimulus set is not {n_categories}."
        assert np.array_equal(
            np.repeat(stimulus_set['stimulus_id'], n_repetitions),
            assembly['stimulus_id'].values
        ), f"Stimulus IDs in Papale2025_stim_{split} stimulus set and assembly do not match."