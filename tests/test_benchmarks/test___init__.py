import os
import pickle

import pytest
from pathlib import Path
from pytest import approx

from brainscore.benchmarks import benchmark_pool, public_benchmark_pool, evaluation_benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures


class TestPoolList:
    """ ensures that the right benchmarks are in the right benchmark pool """

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013.V1-pls',
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Majaj2015public.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_global(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.Majaj2015public.IT-pls',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_public(self, benchmark):
        assert benchmark in public_benchmark_pool

    def test_exact_evaluation_pool(self):
        assert set(evaluation_benchmark_pool.keys()) == {
            'movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls',
            'dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Kar2019-ost',
            'dicarlo.Rajalingham2018-i2n',
            'fei-fei.Deng2009-top1',
        }


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

    @pytest.mark.parametrize('benchmark, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', approx(.686929, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', approx(.573678, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('tolias.Cadena2017-pls', approx(.577474, abs=.005),
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
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-rdm', approx(1, abs=.001),
                     marks=[pytest.mark.memory_intense]),
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
        precomputed_paths = set(map(lstrip_local, precomputed_features['stimulus_path'].values))
        # attach stimulus set meta
        stimulus_set = benchmark._assembly.stimulus_set
        expected_stimulus_paths = list(
            map(lstrip_local, [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]))
        assert (precomputed_paths == set(expected_stimulus_paths))
        for column in stimulus_set.columns:
            precomputed_features[column] = 'presentation', stimulus_set[column].values
        precomputed_features = PrecomputedFeatures(precomputed_features)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected


def lstrip_local(path):
    parts = path.split(os.sep)
    path = os.sep.join(parts[-3:])
    return path
