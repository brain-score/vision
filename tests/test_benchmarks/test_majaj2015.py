import numpy as np
import pytest
from pytest import approx

from brainscore.benchmarks.majaj2015 import DicarloMajaj2015ITMask, DicarloMajaj2015TemporalITPLS, \
    DicarloMajaj2015TemporalV4PLS
from brainscore.benchmarks.majaj2015 import load_assembly
from . import StoredPrecomputedFeatures, check_standard_format


class TestPrecomputed:
    @pytest.mark.requires_gpu
    def test_IT_mask_alexnet(self):
        benchmark = DicarloMajaj2015ITMask()
        candidate = StoredPrecomputedFeatures('alexnet-hvmv6-features.6.pkl')
        score = benchmark(candidate).raw
        assert score.sel(aggregation='center') == approx(.614621, abs=.005)


@pytest.mark.private_access
class TestAssembly:
    def test_majaj2015_V4(self):
        assembly = load_assembly(region='V4')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'V4'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88

    def test_majaj2015_IT(self):
        assembly = load_assembly(region='IT')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168

    @pytest.mark.memory_intense
    def test_majaj2015temporal_V4_from_benchmark(self):
        benchmark = DicarloMajaj2015TemporalV4PLS()
        assembly = benchmark._assembly
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88
        assert len(assembly['time_bin']) == 39
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 231, 20)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(20, 251, 20)))

    @pytest.mark.memory_intense
    def test_majaj2015temporal_IT_from_benchmark(self):
        benchmark = DicarloMajaj2015TemporalITPLS()
        assembly = benchmark._assembly
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168
        np.testing.assert_array_equal(assembly['time_bin_start'], list(range(0, 231, 20)))
        np.testing.assert_array_equal(assembly['time_bin_end'], list(range(20, 251, 20)))
