import os

import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from pytest import approx
from typing import List, Tuple
import xarray as xr

from brainscore.benchmarks import benchmark_pool, public_benchmark_pool, evaluation_benchmark_pool, \
    engineering_benchmark_pool
from brainscore.model_interface import BrainModel
from tests.test_benchmarks import PrecomputedFeatures
from brainio.assemblies import BehavioralAssembly


class TestPoolList:
    """ ensures that the right benchmarks are in the right benchmark pool """

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013.V1-pls',
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.MajajHong2015.IT-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_global(self, benchmark):
        assert benchmark in benchmark_pool

    @pytest.mark.parametrize('benchmark', [
        'movshon.FreemanZiemba2013public.V1-pls',
        'dicarlo.MajajHong2015public.IT-pls',
        'dicarlo.Rajalingham2018public-i2n',
        'fei-fei.Deng2009-top1',
    ])
    def test_contained_public(self, benchmark):
        assert benchmark in public_benchmark_pool

    def test_exact_evaluation_pool(self):
        assert set(evaluation_benchmark_pool.keys()) == {
            # V1
            'movshon.FreemanZiemba2013.V1-pls',
            'dicarlo.Marques2020_Ringach2002-or_bandwidth',
            'dicarlo.Marques2020_Ringach2002-or_selective',
            'dicarlo.Marques2020_Ringach2002-circular_variance',
            'dicarlo.Marques2020_Ringach2002-orth_pref_ratio',
            'dicarlo.Marques2020_Ringach2002-cv_bandwidth_ratio',
            'dicarlo.Marques2020_DeValois1982-pref_or',
            'dicarlo.Marques2020_Ringach2002-opr_cv_diff',
            'dicarlo.Marques2020_Schiller1976-sf_bandwidth',
            'dicarlo.Marques2020_Schiller1976-sf_selective',
            'dicarlo.Marques2020_DeValois1982-peak_sf',
            'dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness',
            'dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity',
            'dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio',
            'dicarlo.Marques2020_Ringach2002-modulation_ratio',
            'dicarlo.Marques2020_Cavanaugh2002-grating_summation_field',
            'dicarlo.Marques2020_Cavanaugh2002-surround_diameter',
            'dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index',
            'dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index',
            'dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
            'dicarlo.Marques2020_FreemanZiemba2013-max_noise',
            'dicarlo.Marques2020_FreemanZiemba2013-max_texture',
            'dicarlo.Marques2020_Ringach2002-max_dc',
            # V2
            'movshon.FreemanZiemba2013.V2-pls',
            # V4
            'dicarlo.MajajHong2015.V4-pls',
            'dicarlo.Sanghavi2020.V4-pls',
            'dicarlo.SanghaviJozwik2020.V4-pls',
            'dicarlo.SanghaviMurty2020.V4-pls',
            # IT
            'dicarlo.MajajHong2015.IT-pls',
            'dicarlo.Sanghavi2020.IT-pls',
            'dicarlo.SanghaviJozwik2020.IT-pls',
            'dicarlo.SanghaviMurty2020.IT-pls',
            'dicarlo.Kar2019-ost',
            # behavior
            'dicarlo.Rajalingham2018-i2n',
        }

    def test_engineering_pool(self):
        assert set(engineering_benchmark_pool.keys()) == {
            'fei-fei.Deng2009-top1',
            'dietterich.Hendrycks2019-noise-top1', 'dietterich.Hendrycks2019-blur-top1',
            'dietterich.Hendrycks2019-weather-top1', 'dietterich.Hendrycks2019-digital-top1'
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
        pytest.param('dicarlo.MajajHong2015.V4-pls', approx(.89503, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-pls', approx(.821841, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.V4-rdm', approx(.936473, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-rdm', approx(.887618, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Sanghavi2020.V4-pls', approx(.8892049, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Sanghavi2020.IT-pls', approx(.868293, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviJozwik2020.V4-pls', approx(.9630336, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviJozwik2020.IT-pls', approx(.860352, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviMurty2020.V4-pls', approx(.9666086, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviMurty2020.IT-pls', approx(.875714, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Rajalingham2020.IT-pls', approx(.561013, abs=.001),
                     marks=[pytest.mark.memory_intense, pytest.mark.slow]),
        # V1 properties Ceilings
        pytest.param('dicarlo.Marques2020_DeValois1982-pref_or', approx(0.962, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-circular_variance', approx(0.959, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-or_bandwidth', approx(0.964, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-orth_pref_ratio', approx(0.962, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-or_selective', approx(0.994, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-cv_bandwidth_ratio', approx(0.968, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-opr_cv_diff', approx(0.967, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_DeValois1982-peak_sf', approx(0.967, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Schiller1976-sf_selective', approx(0.963, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Schiller1976-sf_bandwidth', approx(0.933, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Cavanaugh2002-grating_summation_field', approx(0.956, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Cavanaugh2002-surround_diameter', approx(0.955, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index', approx(0.958, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index', approx(0.948, abs=.005),
                     marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(0.958, abs=.005),
                     marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-modulation_ratio', approx(0.959, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity', approx(0.940, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness', approx(0.935, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(0.939, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_Ringach2002-max_dc', approx(0.968, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-max_texture', approx(0.946, abs=.005), marks=[]),
        pytest.param('dicarlo.Marques2020_FreemanZiemba2013-max_noise', approx(0.945, abs=.005), marks=[]),
    ])
    def test_ceilings(self, benchmark, expected):
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == expected

    @pytest.mark.parametrize('benchmark, visual_degrees, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 4, approx(.668491, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 4, approx(.553155, abs=.001),
                     marks=[pytest.mark.memory_intense]),
        pytest.param('tolias.Cadena2017-pls', 2, approx(.577474, abs=.005),
                     marks=pytest.mark.private_access),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 8, approx(.923713, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 8, approx(.823433, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Sanghavi2020.V4-pls', 8, approx(.9727137, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Sanghavi2020.IT-pls', 8, approx(.890062, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviJozwik2020.V4-pls', 8, approx(.9739177, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviJozwik2020.IT-pls', 8, approx(.9999779, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviMurty2020.V4-pls', 5, approx(.978581, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.SanghaviMurty2020.IT-pls', 5, approx(.9997532, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.Rajalingham2020.IT-pls', 8, approx(.693463, abs=.005),
                     marks=[pytest.mark.memory_intense, pytest.mark.slow]),
    ])
    def test_self_regression(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
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
        pytest.param('dicarlo.MajajHong2015.V4-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
        pytest.param('dicarlo.MajajHong2015.IT-rdm', 8, approx(1, abs=.001),
                     marks=pytest.mark.memory_intense),
    ])
    def test_self_rdm(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_pool[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


@pytest.mark.private_access
class TestPrecomputed:
    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('movshon.FreemanZiemba2013.V1-pls', approx(.466222, abs=.005)),
        ('movshon.FreemanZiemba2013.V2-pls', approx(.459283, abs=.005)),
    ])
    def test_FreemanZiemba2013(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-freemanziemba2013.aperture-private.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.MajajHong2015.V4-pls', approx(.490236, abs=.005)),
        ('dicarlo.MajajHong2015.IT-pls', approx(.584053, abs=.005)),
    ])
    def test_MajajHong2015(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-majaj2015.private-features.12.nc', expected=expected)

    def run_test(self, benchmark, file, expected):
        benchmark = benchmark_pool[benchmark]
        precomputed_features = Path(__file__).parent / file
        precomputed_features = BehavioralAssembly(xr.load_dataarray(precomputed_features))
        precomputed_features = precomputed_features.stack(presentation=['stimulus_path'])
        precomputed_paths = list(map(lambda f: Path(f).name, precomputed_features['stimulus_path'].values))
        # attach stimulus set meta
        stimulus_set = benchmark._assembly.stimulus_set
        expected_stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
        expected_stimulus_paths = list(map(lambda f: Path(f).name, expected_stimulus_paths))
        assert set(precomputed_paths) == set(expected_stimulus_paths)
        for column in stimulus_set.columns:
            precomputed_features[column] = 'presentation', stimulus_set[column].values
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=10,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected

    @pytest.mark.memory_intense
    @pytest.mark.private_access
    @pytest.mark.slow
    def test_Kar2019ost_cornet_s(self):
        benchmark = benchmark_pool['dicarlo.Kar2019-ost']
        precomputed_features = Path(__file__).parent / 'cornet_s-kar2019.nc'
        precomputed_features = BehavioralAssembly(xr.load_dataarray(precomputed_features))
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == approx(.316, abs=.005)

    def test_Rajalingham2018public(self):
        benchmark = benchmark_pool['dicarlo.Rajalingham2018public-i2n']
        # load features
        precomputed_features = Path(__file__).parent / 'CORnetZ-rajalingham2018public.nc'
        precomputed_features = BehavioralAssembly(xr.load_dataarray(precomputed_features))
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == approx(.136923, abs=.005)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.Sanghavi2020.V4-pls', approx(.551135, abs=.015)),
        ('dicarlo.Sanghavi2020.IT-pls', approx(.611347, abs=.015)),
    ])
    def test_Sanghavi2020(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-sanghavi2020-features.12.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.SanghaviJozwik2020.V4-pls', approx(.49235, abs=.005)),
        ('dicarlo.SanghaviJozwik2020.IT-pls', approx(.590543, abs=.005)),
    ])
    def test_SanghaviJozwik2020(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-sanghavijozwik2020-features.12.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.SanghaviMurty2020.V4-pls', approx(.357461, abs=.015)),
        ('dicarlo.SanghaviMurty2020.IT-pls', approx(.53006, abs=.015)),
    ])
    def test_SanghaviMurty2020(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-sanghavimurty2020-features.12.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.Rajalingham2020.IT-pls', approx(.147549, abs=.01)),
    ])
    def test_Rajalingham2020(self, benchmark, expected):
        self.run_test(benchmark=benchmark, file='alexnet-rajalingham2020-features.12.nc', expected=expected)

    @pytest.mark.memory_intense
    @pytest.mark.slow
    @pytest.mark.parametrize('benchmark, expected', [
        ('dicarlo.Marques2020_DeValois1982-pref_or', approx(.895, abs=.01)),
        ('dicarlo.Marques2020_DeValois1982-pref_or', approx(.895, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-circular_variance', approx(.830, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-or_bandwidth', approx(.844, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-orth_pref_ratio', approx(.876, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-or_selective', approx(.895, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-cv_bandwidth_ratio', approx(.841, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-opr_cv_diff', approx(.909, abs=.01)),
        ('dicarlo.Marques2020_DeValois1982-peak_sf', approx(.775, abs=.01)),
        ('dicarlo.Marques2020_Schiller1976-sf_selective', approx(.808, abs=.01)),
        ('dicarlo.Marques2020_Schiller1976-sf_bandwidth', approx(.869, abs=.01)),
        ('dicarlo.Marques2020_Cavanaugh2002-grating_summation_field', approx(.599, abs=.01)),
        ('dicarlo.Marques2020_Cavanaugh2002-surround_diameter', approx(.367, abs=.01)),
        ('dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index', approx(.365, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index', approx(.636, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index', approx(.861, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-modulation_ratio', approx(.371, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity', approx(.646, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness', approx(.508, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio', approx(.827, abs=.01)),
        ('dicarlo.Marques2020_Ringach2002-max_dc', approx(.904, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-max_texture', approx(.823, abs=.01)),
        ('dicarlo.Marques2020_FreemanZiemba2013-max_noise', approx(.684, abs=.01)),
    ])
    def test_Marques2020(self, benchmark, expected):
        self.run_test_properties(
            benchmark=benchmark,
            files={'dicarlo.Marques2020_blank': 'alexnet-dicarlo.Marques2020_blank.nc',
                   'dicarlo.Marques2020_receptive_field': 'alexnet-dicarlo.Marques2020_receptive_field.nc',
                   'dicarlo.Marques2020_orientation': 'alexnet-dicarlo.Marques2020_orientation.nc',
                   'dicarlo.Marques2020_spatial_frequency': 'alexnet-dicarlo.Marques2020_spatial_frequency.nc',
                   'dicarlo.Marques2020_size': 'alexnet-dicarlo.Marques2020_size.nc',
                   'movshon.FreemanZiemba2013_properties': 'alexnet-movshon.FreemanZiemba2013_properties.nc',
                   },
            expected=expected)

    def run_test_properties(self, benchmark, files, expected):
        benchmark = benchmark_pool[benchmark]
        from brainscore import get_stimulus_set

        stimulus_identifiers = np.unique(np.array(['dicarlo.Marques2020_blank', 'dicarlo.Marques2020_receptive_field',
                                                   'dicarlo.Marques2020_orientation',
                                                   benchmark._assembly.stimulus_set.identifier]))
        precomputed_features = {}
        for current_stimulus in stimulus_identifiers:
            stimulus_set = get_stimulus_set(current_stimulus)
            path = Path(__file__).parent / files[current_stimulus]
            features = xr.load_dataarray(path)
            features = BehavioralAssembly(features).stack(presentation=['stimulus_path'])
            precomputed_features[current_stimulus] = features
            precomputed_paths = [Path(f).name for f in precomputed_features[current_stimulus]['stimulus_path'].values]
            # attach stimulus set meta
            expected_stimulus_paths = [stimulus_set.get_image(image_id) for image_id in stimulus_set['image_id']]
            expected_stimulus_paths = list(map(lambda f: Path(f).name, expected_stimulus_paths))
            assert set(precomputed_paths) == set(expected_stimulus_paths)
            for column in stimulus_set.columns:
                precomputed_features[current_stimulus][column] = 'presentation', stimulus_set[column].values

        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected


class TestVisualDegrees:
    @pytest.mark.parametrize('benchmark, candidate_degrees, image_id, expected', [
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V1-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013public.V1-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.314561, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013public.V1-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.23113, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 14, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.31429, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013.V2-pls', 6, 'c3a633a13e736394f213ddf44bf124fe80cabe07',
                     approx(.22966, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('movshon.FreemanZiemba2013public.V2-pls', 14, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.314561, abs=.0001), marks=[]),
        pytest.param('movshon.FreemanZiemba2013public.V2-pls', 6, '21041db1f26c142812a66277c2957fb3e2070916',
                     approx(.23113, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015.V4-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015public.V4-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.25071, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015public.V4-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.00460, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 14, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.251345, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015.IT-pls', 6, '40a786ed8e13db10185ddfdbe07759d83a589e1c',
                     approx(.0054886, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.MajajHong2015public.IT-pls', 14, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.25071, abs=.0001), marks=[]),
        pytest.param('dicarlo.MajajHong2015public.IT-pls', 6, '8a72e2bfdb8c267b57232bf96f069374d5b21832',
                     approx(.00460, abs=.0001), marks=[]),
        pytest.param('dicarlo.Kar2019-ost', 14, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                     approx(.225021, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Kar2019-ost', 6, '6d19b24c29832dfb28360e7731e3261c13a4287f',
                     approx(.001248, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018-i2n', 14, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                     approx(.225023, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018-i2n', 6, '0223bf9e5db0edad21976b16494fe9396a5ef145',
                     approx(.002244, abs=.0001), marks=[pytest.mark.private_access]),
        pytest.param('dicarlo.Rajalingham2018public-i2n', 14, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                     approx(.22486, abs=.0001), marks=[]),
        pytest.param('dicarlo.Rajalingham2018public-i2n', 6, '0020cef91bd626e9fbbabd853494ee444e5c9ecb',
                     approx(.00097, abs=.0001), marks=[]),
        pytest.param('tolias.Cadena2017-pls', 14, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.32655, abs=.0001),
                     marks=[pytest.mark.private_access]),
        pytest.param('tolias.Cadena2017-pls', 6, '0fe27ddd5b9ea701e380063dc09b91234eba3551', approx(.29641, abs=.0001),
                     marks=[pytest.mark.private_access]),
    ])
    def test_amount_gray(self, benchmark, candidate_degrees, image_id, expected):
        benchmark = benchmark_pool[benchmark]

        class DummyCandidate(BrainModel):
            class StopException(Exception):
                pass

            def visual_degrees(self):
                return candidate_degrees

            def look_at(self, stimuli, number_of_trials=1):
                image = stimuli.get_image(image_id)
                image = Image.open(image)
                image = np.array(image)
                amount_gray = 0
                for index in np.ndindex(image.shape[:2]):
                    color = image[index]
                    gray = [128, 128, 128]
                    if (color == gray).all():
                        amount_gray += 1
                assert amount_gray / image.size == expected
                raise self.StopException()

            def start_task(self, task: BrainModel.Task, fitting_stimuli):
                pass

            def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
                pass

        candidate = DummyCandidate()
        try:
            benchmark(candidate)  # just call to get the stimuli
        except DummyCandidate.StopException:  # but stop early
            pass


class TestNumberOfTrials:
    @pytest.mark.private_access
    @pytest.mark.parametrize('benchmark_identifier', evaluation_benchmark_pool.keys())
    def test_repetitions(self, benchmark_identifier):
        """ Tests that all evaluation benchmarks have repetitions in the stimulus_set """
        benchmark = benchmark_pool[benchmark_identifier]

        class AssertRepeatCandidate(BrainModel):
            class StopException(Exception):
                pass

            def identifier(self) -> str:
                return 'assert-repeat-candidate'

            def visual_degrees(self):
                return 8

            def look_at(self, stimuli, number_of_trials=1):
                assert number_of_trials > 1
                raise self.StopException()

            def start_task(self, task: BrainModel.Task, fitting_stimuli):
                pass

            def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
                pass

        candidate = AssertRepeatCandidate()
        try:
            benchmark(candidate)  # just call to get the stimuli
        except AssertRepeatCandidate.StopException:  # but stop early
            pass


def lstrip_local(path):
    parts = path.split(os.sep)
    brainio_index = parts.index('.brainio')
    path = os.sep.join(parts[brainio_index:])
    return path
