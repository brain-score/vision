import pytest
import brainscore_vision
from brainscore_vision import load_stimulus_set


@pytest.mark.private_access
class TestMarques2020V1Properties:
    @pytest.mark.parametrize('identifier,num_neuroids,properties,stimulus_set_identifier', [
        ('Cavanaugh2002a', 190, [
            'surround_suppression_index', 'strongly_suppressed', 'grating_summation_field', 'surround_diameter',
            'surround_grating_summation_field_ratio'],
         'Marques2020_size'),
        ('DeValois1982a', 385, [
            'preferred_orientation'],
         'Marques2020_orientation'),
        ('DeValois1982b', 363, [
            'peak_spatial_frequency'],
         'Marques2020_spatial_frequency'),
        ('FreemanZiemba2013_V1_properties', 102, [
            'absolute_texture_modulation_index', 'family_variance', 'max_noise', 'max_texture', 'noise_selectivity',
            'noise_sparseness', 'sample_variance', 'texture_modulation_index', 'texture_selectivity',
            'texture_sparseness', 'variance_ratio'],
         'FreemanZiemba2013_properties'),
        ('Ringach2002', 304, [
            'bandwidth', 'baseline', 'circular_variance', 'circular_variance_bandwidth_ratio', 'max_ac', 'max_dc',
            'min_dc', 'modulation_ratio', 'orientation_selective', 'orthogonal_preferred_ratio',
            'orthogonal_preferred_ratio_bandwidth_ratio', 'orthogonal_preferred_ratio_circular_variance_difference'],
         'Marques2020_orientation'),
        ('Schiller1976c', 87, [
            'spatial_frequency_selective', 'spatial_frequency_bandwidth'],
         'Marques2020_spatial_frequency'),
    ])
    def test_assembly(self, identifier, num_neuroids, properties, stimulus_set_identifier):
        assembly = brainscore_vision.load_dataset(identifier)
        assert set(assembly.dims) == {'neuroid', 'neuronal_property'}
        assert len(assembly['neuroid']) == num_neuroids
        assert set(assembly['neuronal_property'].values) == set(properties)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == stimulus_set_identifier


@pytest.mark.private_access
class TestMarques2020V1Properties:
    @pytest.mark.parametrize('identifier,num_stimuli', [
        ('Marques2020_blank', 1),
        ('Marques2020_receptive_field', 3528),
        ('Marques2020_orientation', 1152),
        ('Marques2020_spatial_frequency', 2112),
        ('Marques2020_size', 2304),
        ('FreemanZiemba2013_properties', 450),
    ])
    def test_num_stimuli(self, identifier, num_stimuli):
        stimulus_set = load_stimulus_set(identifier)
        assert len(stimulus_set) == num_stimuli
