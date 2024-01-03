import pytest

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
def test_assembly():
    assembly = load_dataset('Igustibagus2024')
    check_standard_format(assembly, nans_expected=True)
    assert assembly.attrs['stimulus_set_identifier'] == 'Igustibagus2024'
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 197_694
    assert set(assembly['object_style'].values) == {'silhouette', 'sketch', 'cartoon', 'original', 'painting',
                                                    'line_drawing', 'outline', 'convex_hull', 'mosaic',
                                                    'textures', 'skeleton', 'grayscale', 'nan', 'silhouettes', 'edges'}
    assert len(assembly['neuroid']) == 181
    assert set(assembly['animal'].values) == {'Pico', 'Oleo'}
    assert len(set(assembly['background_id'].values)) == 121


@pytest.mark.private_access
def test_stimulus_set():
    stimulus_set = load_stimulus_set('Igustibagus2024')
    assert len(stimulus_set) == 3138
