from pytest import approx

import brainscore
from brainscore.benchmarks.nonregressing import split_assembly, DicarloMajaj2015IT, DicarloMajaj2015V4, ToliasCadena2017
from tests import private_access


class TestSplitAssembly:
    def test_repeat(self):
        assembly = brainscore.get_assembly('dicarlo.Majaj2015')
        splits0 = split_assembly(assembly)
        splits1 = split_assembly(assembly)
        assert len(splits0) == len(splits1)
        assert all(s0.equals(s1) for s0, s1 in zip(splits0, splits1))


class TestMajaj2015:
    def test_ceiling_V4(self):
        benchmark = DicarloMajaj2015V4()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.892, abs=0.01)

    def test_ceiling_IT(self):
        benchmark = DicarloMajaj2015IT()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.817, abs=0.01)

    def test_self(self):
        benchmark = DicarloMajaj2015IT()
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.IT'
        ceiled_score = benchmark(lambda stimuli: source)
        score = ceiled_score.raw
        assert score.sel(aggregation='center') == approx(1)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(0), "too much deviation between neuroids"


@private_access
class TestCadena2017:
    def test_ceiling(self):
        benchmark = ToliasCadena2017()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.577, abs=0.05)

    def test_self(self):
        benchmark = ToliasCadena2017()
        source = benchmark.assembly
        score = benchmark(lambda stimuli: source).raw
        assert score.sel(aggregation='center') == approx(1)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(0)
        assert raw_values.mean('split').std() == approx(0)
