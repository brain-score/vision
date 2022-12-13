import pytest
import brainscore_vision


@pytest.mark.private_access
class TestMarques2020V1Properties:
    @pytest.mark.parametrize('identifier,num_neuroids,properties,stimulus_set_identifier', [
        ('movshon.Cavanaugh2002a', 190, [
            'surround_suppression_index', 'strongly_suppressed', 'grating_summation_field', 'surround_diameter',
            'surround_grating_summation_field_ratio'],
         'dicarlo.Marques2020_size'),
        ('devalois.DeValois1982a', 385, [
            'preferred_orientation'],
         'dicarlo.Marques2020_orientation'),
        ('devalois.DeValois1982b', 363, [
            'peak_spatial_frequency'],
         'dicarlo.Marques2020_spatial_frequency'),
        ('movshon.FreemanZiemba2013_V1_properties', 102, [
            'absolute_texture_modulation_index', 'family_variance', 'max_noise', 'max_texture', 'noise_selectivity',
            'noise_sparseness', 'sample_variance', 'texture_modulation_index', 'texture_selectivity',
            'texture_sparseness', 'variance_ratio'],
         'movshon.FreemanZiemba2013_properties'),
        ('shapley.Ringach2002', 304, [
            'bandwidth', 'baseline', 'circular_variance', 'circular_variance_bandwidth_ratio', 'max_ac', 'max_dc',
            'min_dc', 'modulation_ratio', 'orientation_selective', 'orthogonal_preferred_ratio',
            'orthogonal_preferred_ratio_bandwidth_ratio', 'orthogonal_preferred_ratio_circular_variance_difference'],
         'dicarlo.Marques2020_orientation'),
        ('schiller.Schiller1976c', 87, [
            'spatial_frequency_selective', 'spatial_frequency_bandwidth'],
         'dicarlo.Marques2020_spatial_frequency'),
    ])
    def test_assembly(self, identifier, num_neuroids, properties, stimulus_set_identifier):
        assembly = brainscore_vision.get_assembly(identifier)
        assert set(assembly.dims) == {'neuroid', 'neuronal_property'}
        assert len(assembly['neuroid']) == num_neuroids
        assert set(assembly['neuronal_property'].values) == set(properties)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == stimulus_set_identifier
