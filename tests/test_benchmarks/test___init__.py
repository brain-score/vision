import pickle

from pathlib import Path

import pytest
from pytest import approx

from brainscore.benchmarks import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures


@pytest.mark.private_access
class TestStandardized:
    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.873345, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.824836, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', approx(.918672, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(.856968, abs=.001),
                     marks=[pytest.mark.memory_intense]),
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

    @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 4, approx(.686929, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 4, approx(.573678, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('tolias.Cadena2017-pls', 2, approx(.577474, abs=.005),
                     marks=pytest.mark.private_access),
        pytest.param('dicarlo.Majaj2015.V4-pls', 8, approx(.923713, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-pls', 8, approx(.823433, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_regression(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'neuroid')
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10

    @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-rdm', 4, approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', 4, approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('dicarlo.Majaj2015.V4-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Majaj2015.IT-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_rdm(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        source = benchmark._assembly
        score = benchmark(PrecomputedFeatures(source, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


@pytest.mark.private_access
class TestPrecomputed:
    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('movshon.FreemanZiemba2013.V1-pls', approx(.326559, abs=.005)),
        ('movshon.FreemanZiemba2013.V2-pls', approx(.419765, abs=.005)),
    ])
    def test_FreemanZiemba2013(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-freemanziemba2013.private-features.12.pkl', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.Majaj2015.V4-pls', approx(.490236, abs=.005)),
        ('dicarlo.Majaj2015.IT-pls', approx(.584053, abs=.005)),
    ])
    def test_Majaj2015(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-majaj2015.private-features.12.pkl', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.requires_gpu
    def test_IT_mask_alexnet(self):
        self.run_test(benchmark='dicarlo.Majaj2015.IT-mask',
                      file='alexnet-majaj2015.private-features.12.pkl',
                      expected=approx(.594399, abs=.005))

    def run_test(self, benchmark, file, expected):
        benchmark = benchmark_pool[benchmark]
        precomputed_features = Path(__file__).parent / file
        with open(precomputed_features, 'rb') as f:
            precomputed_features = pickle.load(f)['data']
        precomputed_features = precomputed_features.stack(presentation=['stimulus_path'])
        # attach stimulus set meta
        stimulus_set = benchmark._assembly.stimulus_set
        expected_stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
        assert (precomputed_features['stimulus_path'].values == expected_stimulus_paths).all()
        for column in stimulus_set.columns:
            precomputed_features[column] = 'presentation', stimulus_set[column].values
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=10,  # doesn't matter
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected
