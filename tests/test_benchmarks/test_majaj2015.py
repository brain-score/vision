import numpy as np
import pytest
from pathlib import Path
from pytest import approx

from brainscore.benchmarks.majaj2015 import DicarloMajaj2015ITMask
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
        assembly = load_assembly(average_repetitions=True, region='V4')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'V4'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 88

    def test_majaj2015_IT(self):
        assembly = load_assembly(average_repetitions=True, region='IT')
        check_standard_format(assembly)
        assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-private'
        assert set(assembly['region'].values) == {'IT'}
        assert len(assembly['presentation']) == 2560
        assert len(assembly['neuroid']) == 168
