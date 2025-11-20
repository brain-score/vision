import pytest
import numpy as np
from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers.neural_common import filter_reliable_neuroids

TOTAL_VOXELS = 18394
NOISE_CEILING_THRESHOLD = 0.3 * 100
EXPECTED_RELIABLE_VOXEL_COUNTS = {
    '01': {'IT': 250, 'V1': 385, 'V2': 257, 'V4': 152},
    '02': {'IT': 389, 'V1': 356, 'V2': 278, 'V4': 225},
    '03': {'IT': 120, 'V1': 174, 'V2': 131, 'V4': 101},
}
TOTAL_RELIABLE_VOXELS = sum(
    sum(region_counts.values()) for region_counts in EXPECTED_RELIABLE_VOXEL_COUNTS.values()
)

class TestAssemblyProperties:
    @pytest.mark.parametrize('assembly', ['Hebart2023_fmri_train', 'Hebart2023_fmri_test'])
    def test_assembly_existence(self, assembly):
        assert load_dataset(assembly) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 720, 12, 1),
        ('test', 100, 1, 12),
    ])
    def test_assembly_shape(self, split, n_categories, n_examples, n_repetitions):
        assembly = load_dataset(f'Hebart2023_fmri_{split}')
        # assembly shape:
        # (n_categories * n_examples * n_repetitions, n_voxels, 1)
        assert assembly.shape == (n_categories * n_examples * n_repetitions, TOTAL_VOXELS, 1), f"Got shape {assembly.shape}"
        
        reliable_assembly = filter_reliable_neuroids(assembly, NOISE_CEILING_THRESHOLD, 'nc_testset')
        for subject in EXPECTED_RELIABLE_VOXEL_COUNTS.keys():
            for region in EXPECTED_RELIABLE_VOXEL_COUNTS[subject].keys():
                assert len(reliable_assembly.sel(subject=subject).sel(region=region)['neuroid_id']) == EXPECTED_RELIABLE_VOXEL_COUNTS[subject][region], \
                    f"Number of neuroids for subject {subject} in region {region} does not match expected."
        
        # Check all expected coords exist
        expected_presentation_coords = ['stimulus_id', 'stimulus_label', 'object_name', 'stimulus_label_idx', 'repetition']
        expected_neuroid_coords = ['region', 'subject', 'neuroid_id',
                                   'voxel_id', 'voxel_x', 'voxel_y', 'voxel_z',
                                   'nc_singletrial', 'nc_testset',
                                   'splithalf_uncorrected', 'splithalf_corrected', 
                                   'prf-eccentricity', 'prf-polarangle', 'prf-rsquared', 'prf-size', 
                                   'roi']
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
    @pytest.mark.parametrize('stim_set', ['Hebart2023_fmri_stim_train', 'Hebart2023_fmri_stim_test'])
    def test_stimulus_set_existence(self, stim_set):
        assert load_stimulus_set(stim_set) is not None

    @pytest.mark.parametrize('split,n_categories,n_examples,n_repetitions', [
        ('train', 720, 12, 1),
        ('test', 100, 1, 12),
    ])
    def test_stimulus_contents(self, split, n_categories, n_examples, n_repetitions):
        stimulus_set = load_stimulus_set(f'Hebart2023_fmri_stim_{split}')
        assembly = load_dataset(f'Hebart2023_fmri_{split}')
        assert len(stimulus_set) == n_categories * n_examples, f"Got length {len(stimulus_set)}"
        assert len(stimulus_set['label'].unique()) == n_categories, f"Number of unique labels in stimulus set is not {n_categories}."
        assert np.array_equal(
            np.repeat(stimulus_set['stimulus_id'], n_repetitions),
            assembly['stimulus_id'].values
        ), f"Stimulus IDs in Hebart2023_fmri_stim_{split} stimulus set and assembly do not match."