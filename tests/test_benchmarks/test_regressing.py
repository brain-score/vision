import os
import pickle

from pytest import approx

from brainscore.benchmarks.regressing import DicarloMajaj2015ITPLS, DicarloMajaj2015V4PLS, \
    MovshonFreemanZiemba2013V1PLS, MovshonFreemanZiemba2013V2PLS, DicarloMajaj2015ITMask
from brainscore.model_interface import BrainModel


class TestMajaj2015:
    def test_V4_PLS_self(self):
        benchmark = DicarloMajaj2015V4PLS()
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.V4'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.923787, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.002602, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.138202, abs=.00001), "too much deviation between neuroids"

    def test_IT_PLS_self(self):
        benchmark = DicarloMajaj2015ITPLS()
        source = benchmark.assembly
        source.name = 'dicarlo.Majaj2015.IT'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.823865, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.006581, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.157708, abs=.00001), "too much deviation between neuroids"

    def test_IT_mask_alexnet(self):
        benchmark = DicarloMajaj2015ITMask()
        candidate = PrecomputedFeatures('alexnet-hvmv6-features.6.pkl')
        score = benchmark(candidate).raw
        assert score.sel(aggregation='center') == approx(.614621, abs=.005)


class TestMovshonFreemanZiemba2013:
    def test_V1_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V1PLS()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.725142, abs=.0001)

    def test_V2_ceiling(self):
        benchmark = MovshonFreemanZiemba2013V2PLS()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == approx(.13475, abs=.0001)

    def test_V1_self(self):
        benchmark = MovshonFreemanZiemba2013V1PLS()
        source = benchmark.assembly
        source.name = 'movshon.FreemanZiemba2013.V1'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.53432, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.028397, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.358114, abs=.00001), "too much deviation between neuroids"

    def test_V2_self(self):
        benchmark = MovshonFreemanZiemba2013V2PLS()
        source = benchmark.assembly
        source.name = 'movshon.FreemanZiemba2013.V2'
        score = benchmark(lambda *_: source).raw
        assert score.sel(aggregation='center') == approx(.179453, abs=.00001)
        raw_values = score.attrs['raw']
        assert raw_values.median('neuroid').std() == approx(.022404, abs=.00001), "too much deviation between splits"
        assert raw_values.mean('split').std() == approx(.397778, abs=.00001), "too much deviation between neuroids"


class PrecomputedFeatures(BrainModel):
    def __init__(self, filename):
        self.filepath = os.path.join(os.path.dirname(__file__), filename)
        assert os.path.isfile(self.filepath)

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli):
        with open(self.filepath, 'rb') as f:
            features = pickle.load(f)['data']
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return features
