from pathlib import Path

import numpy as np
import pytest
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from brainscore.benchmarks.geirhos2021 import DATASETS
from tests.test_benchmarks import PrecomputedFeatures


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 12 + 5

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_pool(self, dataset):
        identifier = f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"
        assert identifier in benchmark_pool

    def test_mean_ceiling(self):
        benchmarks = [f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency" for dataset in DATASETS]
        benchmarks = [benchmark_pool[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling for benchmark in benchmarks]
        mean_ceiling = np.mean(ceilings)
        assert mean_ceiling == approx(0.43122, abs=0.001)

    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('colour', approx(0.41543, abs=0.001)),
        ('contrast', approx(0.43703, abs=0.001)),
        ('cue-conflict', approx(0.33105, abs=0.001)),
        ('edge', approx(0.31844, abs=0.001)),
        ('eidolonI', approx(0.38634, abs=0.001)),
        ('eidolonII', approx(0.45402, abs=0.001)),
        ('eidolonIII', approx(0.45953, abs=0.001)),
        ('false-colour', approx(0.44405, abs=0.001)),
        ('high-pass', approx(0.44014, abs=0.001)),
        ('low-pass', approx(0.46888, abs=0.001)),
        ('phase-scrambling', approx(0.44667, abs=0.001)),
        ('power-equalisation', approx(0.51063, abs=0.001)),
        ('rotation', approx(0.43851, abs=0.001)),
        ('silhouette', approx(0.47571, abs=0.001)),
        ('sketch', approx(0.36962, abs=0.001)),
        ('stylized', approx(0.50058, abs=0.001)),
        ('uniform-noise', approx(0.43406, abs=0.001)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"
        benchmark = benchmark_pool[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.values.item() == expected_ceiling

    @pytest.mark.parametrize('dataset, model, expected_raw_score', [
        ('colour', 'resnet-50-pytorch', approx(0.21135, abs=0.001)),
        ('contrast', 'resnet-50-pytorch', approx(0.22546, abs=0.001)),
        ('cue-conflict', 'resnet-50-pytorch', approx(0.06800, abs=0.001)),
        ('edge', 'resnet-50-pytorch', approx(0.03626, abs=0.001)),
        ('eidolonI', 'resnet-50-pytorch', approx(0.21029, abs=0.001)),
        ('eidolonII', 'resnet-50-pytorch', approx(0.25090, abs=0.001)),
        ('eidolonIII', 'resnet-50-pytorch', approx(0.15806, abs=0.001)),
        ('false-colour', 'resnet-50-pytorch', approx(0.29431, abs=0.001)),
        ('high-pass', 'resnet-50-pytorch', approx(0.20300, abs=0.001)),
        ('low-pass', 'resnet-50-pytorch', approx(0.16741, abs=0.001)),
        ('phase-scrambling', 'resnet-50-pytorch', approx(0.26879, abs=0.001)),
        ('power-equalisation', 'resnet-50-pytorch', approx(0.29293, abs=0.001)),
        ('rotation', 'resnet-50-pytorch', approx(0.16892, abs=0.001)),
        ('silhouette', 'resnet-50-pytorch', approx(0.42805, abs=0.001)),
        ('sketch', 'resnet-50-pytorch', approx(0.10537, abs=0.001)),
        ('stylized', 'resnet-50-pytorch', approx(0.25430, abs=0.001)),
        ('uniform-noise', 'resnet-50-pytorch', approx(0.19839, abs=0.001)),
    ])
    def test_model_3degrees(self, dataset, model, expected_raw_score):
        benchmark = benchmark_pool[f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features,
                                                   visual_degrees=8,  # doesn't matter, features are already computed
                                                   )
        # score
        score = benchmark(precomputed_features)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score.sel(aggregation='center') >= raw_score.sel(aggregation='center')
        assert raw_score.sel(aggregation='center') == expected_raw_score

    @pytest.mark.parametrize('model, expected_raw_score', [
        ('resnet-50-pytorch-3deg', approx(0.20834, abs=0.001)),
        ('resnet-50-pytorch-8deg', approx(0.10102, abs=0.001)),
    ])
    def test_model_mean(self, model, expected_raw_score):
        scores = []
        for dataset in DATASETS:
            benchmark = benchmark_pool[f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"]
            precomputed_features = Path(__file__).parent / f'{model}-Geirhos2021_{dataset}.nc'
            precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
            precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
            score = benchmark(precomputed_features).raw
            scores.append(score.sel(aggregation='center'))
        mean_score = np.mean(scores)
        assert mean_score == expected_raw_score


class TestEngineering:
    @pytest.mark.parametrize('dataset, model, expected_accuracy', [
        ('colour', 'resnet-50-pytorch', approx(0.96875, abs=0.001)),
        ('contrast', 'resnet-50-pytorch', approx(0.81625, abs=0.001)),
        ('cue-conflict', 'resnet-50-pytorch', approx(0.18203, abs=0.001)),
        ('edge', 'resnet-50-pytorch', approx(0.13750, abs=0.001)),
        ('eidolonI', 'resnet-50-pytorch', approx(0.53375, abs=0.001)),
        ('eidolonII', 'resnet-50-pytorch', approx(0.56719, abs=0.001)),
        ('eidolonIII', 'resnet-50-pytorch', approx(0.58542, abs=0.001)),
        ('false-colour', 'resnet-50-pytorch', approx(0.94286, abs=0.001)),
        ('high-pass', 'resnet-50-pytorch', approx(0.33437, abs=0.001)),
        ('low-pass', 'resnet-50-pytorch', approx(0.43875, abs=0.001)),
        ('phase-scrambling', 'resnet-50-pytorch', approx(0.62031, abs=0.001)),
        ('power-equalisation', 'resnet-50-pytorch', approx(0.73214, abs=0.001)),
        ('rotation', 'resnet-50-pytorch', approx(0.68438, abs=0.001)),
        ('silhouette', 'resnet-50-pytorch', approx(0.54375, abs=0.001)),
        ('sketch', 'resnet-50-pytorch', approx(0.59625, abs=0.001)),
        ('stylized', 'resnet-50-pytorch', approx(0.37125, abs=0.001)),
        ('uniform-noise', 'resnet-50-pytorch', approx(0.44500, abs=0.001)),
    ])
    def test_accuracy(self, dataset, model, expected_accuracy):
        benchmark = benchmark_pool[f"brendel.Geirhos2021{dataset.replace('-', '')}-top1"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        # score
        score = benchmark(precomputed_features)
        assert score.sel(aggregation='center') == expected_accuracy
