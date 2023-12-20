import pytest

from brainscore.benchmarks.domain_transfer_neural import load_domain_transfer
from . import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestAssembly:
    def test_IT(self):
        assembly = load_domain_transfer(average_repetitions=True)
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_identifier'] == 'Igustibagus2024'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 780
        assert set(assembly['object_style'].values) == {'silhouette', 'sketch', 'cartoon', 'original', 'painting', 'line_drawing', 'outline', 'convex_hull', 'mosaic'}
        assert len(assembly['neuroid']) == 110
        assert set(assembly['animal'].values) == {'Pico', 'Oleo'}
        assert assembly['background_id'].values is not None

    