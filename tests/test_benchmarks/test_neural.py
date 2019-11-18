import pytest
from pytest import approx

from brainscore.benchmarks import benchmark_pool
from brainscore.benchmarks.majaj2015 import DicarloMajaj2015ITMask
from tests.test_benchmarks import PrecomputedFeatures, StoredPrecomputedFeatures


class TestStandardized:
    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.873345, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.824836, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', approx(.918672, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(.856968, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('dicarlo.Majaj2015.V4-pls', approx(.89503, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-pls', approx(.821841, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.V4-rdm', approx(.936473, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-rdm', approx(.887618, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_ceilings(self, benchmark, expected):
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == expected

    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.686929, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.573678, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('tolias.Cadena2017-pls', approx(.577474, abs=.001),
                     marks=pytest.mark.private_access),
        pytest.param('dicarlo.Majaj2015.V4-pls', approx(.923713, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-pls', approx(.823433, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_regression(self, benchmark, expected):
        benchmark = benchmark_pool[benchmark]
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'neuroid')
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10

    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.private_access]),
        pytest.param('dicarlo.Majaj2015.V4-rdm', approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-rdm', approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_rdm(self, benchmark, expected):
        benchmark = benchmark_pool[benchmark]
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


class TestPrecomputed:
    @pytest.mark.requires_gpu
    def test_IT_mask_alexnet(self):
        benchmark = DicarloMajaj2015ITMask()
        candidate = StoredPrecomputedFeatures('alexnet-hvmv6-features.6.pkl')
        score = benchmark(candidate).raw
        assert score.sel(aggregation='center') == approx(.614621, abs=.005)
