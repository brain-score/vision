from pytest import approx

from brainscore.benchmarks.regressing import DicarloMajaj2015IT, DicarloMajaj2015V4, MovshonFreemanZiemba2013V1, \
    MovshonFreemanZiemba2013V2


class TestMajaj2015:
    def test_V4_self(self):
        benchmark = DicarloMajaj2015V4()
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.V4'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.923787, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.002602, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.138202, abs=.00001), "too much deviation between neuroids"

    def test_IT_self(self):
        benchmark = DicarloMajaj2015IT()
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.IT'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.823865, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.006581, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.157708, abs=.00001), "too much deviation between neuroids"


class TestMovshonFreemanZiemba2013:
    def test_V1_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V1()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.725142, abs=.0001)

    def test_V2_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V2()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.13475, abs=.0001)

    def test_V1_self(self):
        benchmark = MovshonFreemanZiemba2013V1()
        source = benchmark.assembly
        source.name = 'movshon.FreemanZiemba2013.V1'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.53432, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.028397, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.358114, abs=.00001), "too much deviation between neuroids"

    def test_V2_self(self):
        benchmark = MovshonFreemanZiemba2013V2()
        source = benchmark.assembly
        source.name = 'movshon.FreemanZiemba2013.V2'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.179453, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.022404, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.397778, abs=.00001), "too much deviation between neuroids"
