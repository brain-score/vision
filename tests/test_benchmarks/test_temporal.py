from pytest import approx

from brainscore.benchmarks.temporal import DicarloMajaj2015TemporalV4, DicarloMajaj2015TemporalIT
from tests.flags import private_access, memory_intense
from tests.test_benchmarks import PrecomputedFeatures


@memory_intense
@private_access
class TestMajaj2015:
    def test_V4_self(self):
        benchmark = DicarloMajaj2015TemporalV4()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.temporal.V4'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.710267, abs=.00001)
        raw_values = score.attrs['raw']
        assert len(raw_values['split']) == 10
        assert len(raw_values['time_bin']) == len(source['time_bin'])

    def test_IT_self(self):
        benchmark = DicarloMajaj2015TemporalIT()
        source = benchmark._assembly
        source.name = 'dicarlo.Majaj2015.temporal.IT'
        score = benchmark(PrecomputedFeatures(source)).raw
        assert score.sel(aggregation='center') == approx(.600235, abs=.00001)
        raw_values = score.attrs['raw']
        assert len(raw_values['split']) == 10
        assert len(raw_values['time_bin']) == len(source['time_bin'])
