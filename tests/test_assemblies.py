import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from pytest import approx

import brainio


@pytest.mark.parametrize('assembly', (
        'dicarlo.MajajHong2015',
        'dicarlo.MajajHong2015.private',
        'dicarlo.MajajHong2015.public',
        'dicarlo.MajajHong2015.temporal',
        'dicarlo.MajajHong2015.temporal.private',
        'dicarlo.MajajHong2015.temporal.public',
        'dicarlo.MajajHong2015.temporal-10ms',
        'gallant.David2004',
        'tolias.Cadena2017',
        'movshon.FreemanZiemba2013',
        'movshon.FreemanZiemba2013.private',
        'movshon.FreemanZiemba2013.public',
        'dicarlo.Rajalingham2018.public', 'dicarlo.Rajalingham2018.private',
        'dicarlo.Kar2019',
        'dicarlo.Kar2018hvm',
        'dicarlo.Kar2018cocogray',
        'klab.Zhang2018search_obj_array',
        'aru.Kuzovkin2018',
        'dicarlo.Rajalingham2020',
        'dicarlo.SanghaviMurty2020',
        'dicarlo.SanghaviJozwik2020',
        'dicarlo.Sanghavi2020',
        'dicarlo.SanghaviMurty2020THINGS1',
        'dicarlo.SanghaviMurty2020THINGS2',
        'aru.Kuzovkin2018',
        'dicarlo.Seibert2019',
        'aru.Cichy2019',
        'dicarlo.Rust2012.single',
        'dicarlo.Rust2012.array',
        'dicarlo.BashivanKar2019.naturalistic',
        'dicarlo.BashivanKar2019.synthetic',
        'movshon.Cavanaugh2002a',
        'devalois.DeValois1982a',
        'devalois.DeValois1982b',
        'movshon.FreemanZiemba2013_V1_properties',
        'shapley.Ringach2002',
        'schiller.Schiller1976c',
        'Geirhos2021_colour',
        'Geirhos2021_contrast',
        'Geirhos2021_sketch'
))
def test_list_assembly(assembly):
    l = brainio.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('gallant.David2004', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015.public', marks=[]),
    pytest.param('dicarlo.MajajHong2015.private', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.MajajHong2015.temporal', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('dicarlo.MajajHong2015.temporal.public', marks=[pytest.mark.memory_intense]),
    pytest.param('dicarlo.MajajHong2015.temporal.private',
                 marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    # pytest.param('dicarlo.MajajHong2015.temporal-10ms', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('tolias.Cadena2017', marks=[pytest.mark.private_access]),
    pytest.param('movshon.FreemanZiemba2013', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.public', marks=[pytest.mark.memory_intense]),
    pytest.param('movshon.FreemanZiemba2013.private', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('dicarlo.Rajalingham2018.public', marks=[]),
    pytest.param('dicarlo.Rajalingham2018.private', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Kar2019', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Kar2018hvm', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Kar2018cocogray', marks=[pytest.mark.private_access]),
    pytest.param('klab.Zhang2018search_obj_array', marks=[pytest.mark.private_access]),
    pytest.param('aru.Kuzovkin2018', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rajalingham2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviMurty2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviJozwik2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Sanghavi2020', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviMurty2020THINGS1', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.SanghaviMurty2020THINGS2', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Seibert2019', marks=[pytest.mark.private_access]),
    pytest.param('aru.Cichy2019', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rust2012.single', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.Rust2012.array', marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.BashivanKar2019.naturalistic', marks=[pytest.mark.private_access]),
    pytest.param('Geirhos2021_colour', marks=[pytest.mark.private_access]),
    pytest.param('Geirhos2021_contrast', marks=[pytest.mark.private_access]),
    pytest.param('Geirhos2021_sketch', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert brainio.get_assembly(assembly_identifier) is not None


@pytest.mark.private_access
def test_klab_Zhang2018search():
    assembly = brainio.get_assembly('klab.Zhang2018search_obj_array')
    assert set(assembly.dims) == {'presentation', 'fixation', 'position'}
    assert len(assembly['presentation']) == 4500
    assert len(set(assembly['image_id'].values)) == 300
    assert len(set(assembly['subjects'].values)) == 15
    assert len(assembly['fixation']) == 8
    assert len(assembly['position']) == 2
    assert assembly.stimulus_set is not None


class TestFreemanZiemba:
    @pytest.mark.parametrize('identifier', [
        pytest.param('movshon.FreemanZiemba2013.public', marks=[]),
        pytest.param('movshon.FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_v1_v2_alignment(self, identifier):
        assembly = brainio.get_assembly(identifier)
        v1 = assembly[{'neuroid': [region == 'V1' for region in assembly['region'].values]}]
        v2 = assembly[{'neuroid': [region == 'V2' for region in assembly['region'].values]}]
        assert len(v1['presentation']) == len(v2['presentation'])
        assert set(v1['image_id'].values) == set(v2['image_id'].values)

    @pytest.mark.parametrize('identifier', [
        pytest.param('movshon.FreemanZiemba2013.public', marks=[]),
        pytest.param('movshon.FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_num_neurons(self, identifier):
        assembly = brainio.get_assembly(identifier)
        assert len(assembly['neuroid']) == 205
        v1 = assembly[{'neuroid': [region == 'V1' for region in assembly['region'].values]}]
        assert len(v1['neuroid']) == 102
        v2 = assembly[{'neuroid': [region == 'V2' for region in assembly['region'].values]}]
        assert len(v2['neuroid']) == 103

    @pytest.mark.parametrize('identifier', [
        pytest.param('movshon.FreemanZiemba2013.public', marks=[]),
        pytest.param('movshon.FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_nonzero(self, identifier):
        assembly = brainio.get_assembly(identifier)
        nonzero = np.count_nonzero(assembly)
        assert nonzero > 0

    @pytest.mark.parametrize('identifier, image_id, expected_amount_gray, ratio_gray', [
        pytest.param('movshon.FreemanZiemba2013.public', '21041db1f26c142812a66277c2957fb3e2070916',
                     31756, .3101171875, marks=[]),
        pytest.param('movshon.FreemanZiemba2013.private', 'bfd26c127f8ba028cc95cdc95f00c45c8884b365',
                     31585, .308447265625, marks=[pytest.mark.private_access]),
    ])
    def test_aperture(self, identifier, image_id, expected_amount_gray, ratio_gray):
        """ test a random image for the correct amount of gray pixels """
        assembly = brainio.get_assembly(identifier)
        stimulus_set = assembly.stimulus_set
        image_path = Path(stimulus_set.get_image(image_id))
        assert image_path.is_file()
        # count number of gray pixels in image
        image = Image.open(image_path)
        image = np.array(image)
        amount_gray = 0
        for index in np.ndindex(image.shape[:2]):
            color = image[index]
            gray = [128, 128, 128]
            if (color == gray).all():
                amount_gray += 1
        assert amount_gray / image.size == approx(ratio_gray, abs=.0001)
        assert amount_gray == expected_amount_gray


class TestSeibert:
    @pytest.mark.private_access
    def test_dims(self):
        assembly = brainio.get_assembly('dicarlo.Seibert2019')
        # neuroid: 258 presentation: 286080 time_bin: 1
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 258
        assert len(assembly['presentation']) == 286080
        assert len(assembly['time_bin']) == 1

    @pytest.mark.private_access
    def test_coords(self):
        assembly = brainio.get_assembly('dicarlo.Seibert2019')
        assert len(set(assembly['image_id'].values)) == 5760
        assert len(set(assembly['neuroid_id'].values)) == 258
        assert len(set(assembly['animal'].values)) == 3
        assert len(set(assembly['region'].values)) == 2
        assert len(set(assembly['variation'].values)) == 3

    @pytest.mark.private_access
    def test_content(self):
        assembly = brainio.get_assembly('dicarlo.Seibert2019')
        assert np.count_nonzero(np.isnan(assembly)) == 19118720
        assert assembly.stimulus_set_identifier == "dicarlo.hvm"
        hvm = assembly.stimulus_set
        assert hvm.shape == (5760, 18)


class TestRustSingle:
    @pytest.mark.private_access
    def test_dims(self):
        assembly = brainio.get_assembly('dicarlo.Rust2012.single')
        # (neuroid: 285, presentation: 1500, time_bin: 1)
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 285
        assert len(assembly['presentation']) == 1500
        assert len(assembly['time_bin']) == 1

    @pytest.mark.private_access
    def test_coords(self):
        assembly = brainio.get_assembly('dicarlo.Rust2012.single')
        assert len(set(assembly['image_id'].values)) == 300
        assert len(set(assembly['neuroid_id'].values)) == 285
        assert len(set(assembly['region'].values)) == 2


class TestRustArray:
    @pytest.mark.private_access
    def test_dims(self):
        assembly = brainio.get_assembly('dicarlo.Rust2012.array')
        # (neuroid: 296, presentation: 53700, time_bin: 6)
        assert assembly.dims == ("neuroid", "presentation", "time_bin")
        assert len(assembly['neuroid']) == 296
        assert len(assembly['presentation']) == 53700
        assert len(assembly['time_bin']) == 6

    @pytest.mark.private_access
    def test_coords(self):
        assembly = brainio.get_assembly('dicarlo.Rust2012.array')
        assert len(set(assembly['image_id'].values)) == 300
        assert len(set(assembly['neuroid_id'].values)) == 296
        assert len(set(assembly['animal'].values)) == 2
        assert len(set(assembly['region'].values)) == 2


@pytest.mark.parametrize('assembly,shape,nans', [
    pytest.param('dicarlo.BashivanKar2019.naturalistic', (24320, 233, 1), 309760, marks=[pytest.mark.private_access]),
    pytest.param('dicarlo.BashivanKar2019.synthetic', (21360, 233, 1), 4319940, marks=[pytest.mark.private_access]),
])
def test_BashivanKar2019(assembly, shape, nans):
    assy = brainio.get_assembly(assembly)
    assert assy.shape == shape
    assert np.count_nonzero(np.isnan(assy)) == nans


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
        assembly = brainio.get_assembly(identifier)
        assert set(assembly.dims) == {'neuroid', 'neuronal_property'}
        assert len(assembly['neuroid']) == num_neuroids
        assert set(assembly['neuronal_property'].values) == set(properties)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == stimulus_set_identifier


class TestGeirhos2021:
    @pytest.mark.parametrize('identifier, num_images, num_subjects, stimulus_set_identifier', [
        ('brendel.Geirhos2021_colour', 1280, 4, 'brendel.Geirhos2021_colour'),
        ('brendel.Geirhos2021_contrast', 1280, 4, 'brendel.Geirhos2021_contrast'),
        ('brendel.Geirhos2021_eidolonI', 1280, 4, 'brendel.Geirhos2021_eidolonI'),
        ('brendel.Geirhos2021_eidolonII', 1280, 4, 'brendel.Geirhos2021_eidolonII'),
        ('brendel.Geirhos2021_eidolonIII', 1280, 4, 'brendel.Geirhos2021_eidolonIII'),
        ('brendel.Geirhos2021_false-colour', 1120, 4, 'brendel.Geirhos2021_false-colour'),
        ('brendel.Geirhos2021_high-pass', 1280, 4, 'brendel.Geirhos2021_high-pass'),
        ('brendel.Geirhos2021_low-pass', 1280, 4, 'brendel.Geirhos2021_low-pass'),
        ('brendel.Geirhos2021_phase-scrambling', 1120, 4, 'brendel.Geirhos2021_phase-scrambling'),
        ('brendel.Geirhos2021_sketch', 800, 7, 'brendel.Geirhos2021_sketch'),

    ])
    def test_assembly(self, identifier, num_images, num_subjects, stimulus_set_identifier):
        assembly = brainio.get_assembly(identifier)
        assembly_length = num_subjects * num_images

        # test stimulus set:
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == stimulus_set_identifier

        # test assembly dims:
        assert set(assembly.dims) == {'presentation'}
        assert len(assembly["presentation"]) == assembly_length

        # test assembly coords
        assert len(assembly['image_id']) == assembly_length
        assert len(assembly['image_lookup_id']) == assembly_length
        assert len(assembly['choice']) == assembly_length
        assert len(assembly['truth']) == assembly_length
        assert len(assembly['condition']) == assembly_length
        assert len(assembly['response_time']) == assembly_length
        assert len(assembly['trial']) == assembly_length
        assert len(assembly['subject']) == assembly_length
        assert len(assembly['session']) == assembly_length

        # make sure there are num_images number of unique images (shown 1 time for each subject)
        assert len(np.unique(assembly['image_lookup_id'].values)) == num_images

        # make sure there are num_subjects number of unique subjects
        assert len(np.unique(assembly['subject'].values)) == num_subjects

        # make sure there are 16 unique object categories (ground truths)
        assert len(np.unique(assembly['truth'].values)) == 16
