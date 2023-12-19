import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.private_access
def test_existence():
    assert load_dataset('Cadena2017') is not None


@pytest.mark.private_access
def test_format():
    assembly = load_dataset('Cadena2017')
    check_standard_format(assembly)


@pytest.mark.private_access
def test_num_items():
    assembly = load_dataset('Cadena2017')
    assert assembly.attrs['stimulus_set_identifier'] == 'tolias.Cadena2017'
    assert len(assembly['presentation']) == 29000
    assert len(assembly['neuroid']) == 166


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('Cadena2017')
    assert len(stimulus_set) == 7249
