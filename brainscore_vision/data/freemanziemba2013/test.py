from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytest import approx

from brainscore_vision import load_dataset
from brainscore_vision.benchmark_helpers import check_standard_format
from brainscore_vision.benchmarks.freemanziemba2013.benchmarks.benchmark import load_assembly


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('FreemanZiemba2013', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
    pytest.param('FreemanZiemba2013.public', marks=[pytest.mark.memory_intense]),
    pytest.param('FreemanZiemba2013.private', marks=[pytest.mark.private_access, pytest.mark.memory_intense]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestAssembly:
    def test_V1(self):
        assembly = load_assembly(region='V1', average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'movshon.FreemanZiemba2013.aperture-private'
        assert set(assembly['region'].values) == {'V1'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102

    def test_V2(self):
        assembly = load_assembly(region='V2', average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'movshon.FreemanZiemba2013.aperture-private'
        assert set(assembly['region'].values) == {'V2'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 103


class TestFreemanZiemba:
    @pytest.mark.parametrize('identifier', [
        pytest.param('FreemanZiemba2013.public', marks=[]),
        pytest.param('FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_v1_v2_alignment(self, identifier):
        assembly = load_dataset(identifier)
        v1 = assembly[{'neuroid': [region == 'V1' for region in assembly['region'].values]}]
        v2 = assembly[{'neuroid': [region == 'V2' for region in assembly['region'].values]}]
        assert len(v1['presentation']) == len(v2['presentation'])
        assert set(v1['stimulus_id'].values) == set(v2['stimulus_id'].values)

    @pytest.mark.parametrize('identifier', [
        pytest.param('FreemanZiemba2013.public', marks=[]),
        pytest.param('FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_num_neurons(self, identifier):
        assembly = load_dataset(identifier)
        assert len(assembly['neuroid']) == 205
        v1 = assembly[{'neuroid': [region == 'V1' for region in assembly['region'].values]}]
        assert len(v1['neuroid']) == 102
        v2 = assembly[{'neuroid': [region == 'V2' for region in assembly['region'].values]}]
        assert len(v2['neuroid']) == 103

    @pytest.mark.parametrize('identifier', [
        pytest.param('FreemanZiemba2013.public', marks=[]),
        pytest.param('FreemanZiemba2013.private', marks=[pytest.mark.private_access]),
    ])
    def test_nonzero(self, identifier):
        assembly = load_dataset(identifier)
        nonzero = np.count_nonzero(assembly)
        assert nonzero > 0

    @pytest.mark.parametrize('identifier, image_id, expected_amount_gray, ratio_gray', [
        pytest.param('FreemanZiemba2013.public', '21041db1f26c142812a66277c2957fb3e2070916',
                     31756, .3101171875, marks=[]),
        pytest.param('FreemanZiemba2013.private', 'bfd26c127f8ba028cc95cdc95f00c45c8884b365',
                     31585, .308447265625, marks=[pytest.mark.private_access]),
    ])
    def test_aperture(self, identifier, image_id, expected_amount_gray, ratio_gray):
        """ test a random image for the correct amount of gray pixels """
        assembly = load_dataset(identifier)
        stimulus_set = assembly.stimulus_set
        image_path = Path(stimulus_set.get_stimulus(image_id))
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
