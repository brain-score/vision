import pytest
from pytest import approx
import numpy as np

from brainscore.benchmarks.neural import MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS, \
    DicarloMajaj2015ITPLS, DicarloMajaj2015V4PLS, DicarloMajaj2015ITMask, ToliasCadena2017
from tests.test_benchmarks import PrecomputedFeatures, StoredPrecomputedFeatures


class TestMajaj2015:
    def test_V4_PLS_self(self):
        benchmark = DicarloMajaj2015V4PLS()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.V4'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.921713, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.00416566, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.1434765, abs=.00001), "too much deviation between neuroids"

    def test_IT_PLS_self(self):
        benchmark = DicarloMajaj2015ITPLS()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.IT'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.821433, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.005639, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.155962, abs=.00001), "too much deviation between neuroids"

    @pytest.mark.requires_gpu
    def test_IT_mask_alexnet(self):
        benchmark = DicarloMajaj2015ITMask()
        candidate = StoredPrecomputedFeatures('alexnet-hvmv6-features.6.pkl')
        score = benchmark(candidate).raw
        assert score.sel(aggregation='center') == approx(.614621, abs=.005)


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestMovshonFreemanZiemba2013:
    def test_V1_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V1PLS()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.873345, abs=.0001)

    def test_V2_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V2PLS()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.826023, abs=.0001)

    def test_V1_self(self):
        benchmark = MovshonFreemanZiemba2013V1PLS()
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.676264, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.0210868, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.2449174, abs=.00001), "too much deviation between neuroids"

    def test_V2_self(self):
        benchmark = MovshonFreemanZiemba2013V2PLS()
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.560371, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.0324611, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.2668294, abs=.00001), "too much deviation between neuroids"


@pytest.mark.private_access
class TestToliasCadena2017:
    def test_V1_self(self):
        np.random.seed(0)
        benchmark = ToliasCadena2017()
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(0.577474, abs=.00001), \
                score.sel(aggregation='center')
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0.007518, abs=.00001), \
                "too much deviation between splits" + str(raw_values.median('neuroid').std())
        assert raw_values.mean('split').std() == approx(0.213484, abs=.00001), \
                "too much deviation between neuroids" + str(raw_values.mean('split').std())
