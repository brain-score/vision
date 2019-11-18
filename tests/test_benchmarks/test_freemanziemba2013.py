import numpy as np
import pytest

from brainscore.benchmarks.freemanziemba2013 import load_assembly, MovshonFreemanZiemba2013TemporalV1PLS, \
    MovshonFreemanZiemba2013TemporalV2PLS
from . import check_standard_format


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestAssembly:
    def test_V1(self):
        assembly = load_assembly(region='V1')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013-private'
        assert set(assembly['region'].values) == {'V1'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102

    def test_V2(self):
        assembly = load_assembly(region='V2')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'movshon.FreemanZiemba2013-private'
        assert set(assembly['region'].values) == {'V2'}
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 103

    def test_temporal_V1_from_benchmark(self):
        benchmark = MovshonFreemanZiemba2013TemporalV1PLS()
        assembly = benchmark._assembly
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 102
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 291, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(10, 301, 10)))

    def test_temporal_V2_from_benchmark(self):
        benchmark = MovshonFreemanZiemba2013TemporalV2PLS()
        assembly = benchmark._assembly
        assert len(assembly['presentation']) == 315
        assert len(assembly['neuroid']) == 103
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 291, 10)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(10, 301, 10)))
