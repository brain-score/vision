from pathlib import Path

import numpy as np
import pytest
from pytest import approx

import brainscore_vision
from brainio.assemblies import BehavioralAssembly
from brainscore_vision import benchmark_registry, load_stimulus_set, load_dataset
from brainscore_vision.benchmarks.geirhos2021.benchmark import DATASETS, cast_coordinate_type
from todotests.test_benchmarks import PrecomputedFeatures


@pytest.mark.parametrize('assembly', (
    'brendel.Geirhos2021_colour',
    'brendel.Geirhos2021_contrast',
    'brendel.Geirhos2021_cue-conflict',
    'brendel.Geirhos2021_edge',
    'brendel.Geirhos2021_eidolonI',
    'brendel.Geirhos2021_eidolonII',
    'brendel.Geirhos2021_eidolonIII',
    'brendel.Geirhos2021_false-colour',
    'brendel.Geirhos2021_high-pass',
    'brendel.Geirhos2021_low-pass',
    'brendel.Geirhos2021_phase-scrambling',
    'brendel.Geirhos2021_power-equalisation',
    'brendel.Geirhos2021_rotation',
    'brendel.Geirhos2021_silhouette',
    'brendel.Geirhos2021_stylized',
    'brendel.Geirhos2021_sketch',
    'brendel.Geirhos2021_uniform-noise',
))
def test_list_assembly(assembly):
    l = brainscore_vision.list_assemblies()
    assert assembly in l


@pytest.mark.parametrize('assembly_identifier', [
    pytest.param('brendel.Geirhos2021_colour', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_contrast', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_cue-conflict', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_edge', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonI', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonII', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_eidolonIII', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_false-colour', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_high-pass', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_low-pass', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_phase-scrambling', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_power-equalisation', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_rotation', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_silhouette', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_stylized', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_sketch', marks=[pytest.mark.private_access]),
    pytest.param('brendel.Geirhos2021_uniform-noise', marks=[pytest.mark.private_access]),
])
def test_existence(assembly_identifier):
    assert load_dataset(assembly_identifier) is not None


class TestGeirhos2021:

    # test stimulus_set data alignment with assembly:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'condition',
        'truth',
    ])
    def test_stimulus_set_assembly_alignment(self, identifier, field):
        full_name = f"brendel.Geirhos2021_{identifier}"
        assembly = load_dataset(full_name)
        assert assembly.stimulus_set is not None
        assert assembly.stimulus_set.identifier == full_name
        assert set(assembly.stimulus_set[field]) == set(assembly[field].values)

    # test the number of subjects:
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('colour', 4),
        ('contrast', 4),
        ('cue-conflict', 10),
        ('edge', 10),
        ('eidolonI', 4),
        ('eidolonII', 4),
        ('eidolonIII', 4),
        ('false-colour', 4),
        ('high-pass', 4),
        ('low-pass', 4),
        ('phase-scrambling', 4),
        ('power-equalisation', 4),
        ('rotation', 4),
        ('silhouette', 10),
        ('stylized', 5),
        ('sketch', 7),
        ('uniform-noise', 4),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(assembly['subject'].values)) == num_subjects

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('colour', 1280),
        ('contrast', 1280),
        ('cue-conflict', 1280),
        ('edge', 160),
        ('eidolonI', 1280),
        ('eidolonII', 1280),
        ('eidolonIII', 1280),
        ('false-colour', 1120),
        ('high-pass', 1280),
        ('low-pass', 1280),
        ('phase-scrambling', 1120),
        ('power-equalisation', 1120),
        ('rotation', 1280),
        ('silhouette', 160),
        ('stylized', 800),
        ('sketch', 800),
        ('uniform-noise', 1280),
    ])
    def test_num_images(self, identifier, num_images):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(assembly['image_id'].values)) == num_images

    # tests assembly dim for ALL 17 sets:
    @pytest.mark.parametrize('identifier, length', [
        ('colour', 5120),
        ('contrast', 5120),
        ('cue-conflict', 12800),
        ('edge', 1600),
        ('eidolonI', 5120),
        ('eidolonII', 5120),
        ('eidolonIII', 5120),
        ('false-colour', 4480),
        ('high-pass', 5120),
        ('low-pass', 5120),
        ('phase-scrambling', 4480),
        ('power-equalisation', 4480),
        ('rotation', 5120),
        ('silhouette', 1600),
        ('stylized', 4000),
        ('sketch', 5600),
        ('uniform-noise', 5120),
    ])
    def test_length(self, identifier, length):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert len(assembly['presentation']) == length

    # test assembly coords present in ALL 17 sets:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_id_long',
        'choice',
        'truth',
        'condition',
        'response_time',
        'trial',
        'subject',
        'session',
    ])
    def test_fields_present(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)

    # tests assembly coords for the 2 "abnormal" sets:
    @pytest.mark.parametrize('identifier', [
        'edge',
        'silhouette',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_category',
        'truth',
        'image_variation',
        'condition',
    ])
    def test_fields_present_abnormal_sets(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)

    # tests assembly coords for the cue-conflict different set:
    @pytest.mark.parametrize('identifier', [
        'cue-conflict',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'original_image',
        'truth',
        'category',
        'conflict_image',
        'original_image_category',
        'original_image_variation',
        'conflict_image_category',
        'conflict_image_variation',
        'condition',
    ])
    def test_fields_present_cue_conflict(self, identifier, field):
        assembly = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(assembly, field)


# testing stimulus sets
@pytest.mark.slow
class TestGeirhos2021:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'cue-conflict',
        'edge',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'silhouette',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    def test_stimulus_set_exist(self, identifier):
        full_name = f"brendel.Geirhos2021_{identifier}"
        stimulus_set = load_stimulus_set(full_name)
        assert stimulus_set is not None
        assert stimulus_set.identifier == full_name

    # test the number of images
    @pytest.mark.parametrize('identifier, num_images', [
        ('colour', 1280),
        ('contrast', 1280),
        ('cue-conflict', 1280),
        ('edge', 160),
        ('eidolonI', 1280),
        ('eidolonII', 1280),
        ('eidolonIII', 1280),
        ('false-colour', 1120),
        ('high-pass', 1280),
        ('low-pass', 1280),
        ('phase-scrambling', 1120),
        ('power-equalisation', 1120),
        ('rotation', 1280),
        ('silhouette', 160),
        ('stylized', 800),
        ('sketch', 800),
        ('uniform-noise', 1280),
    ])
    def test_num_images(self, identifier, num_images):
        stimulus_set = load_stimulus_set(f"brendel.Geirhos2021_{identifier}")
        assert len(np.unique(stimulus_set['image_id'].values)) == num_images

    # tests stimulus_set coords for the 14 "normal" sets:
    @pytest.mark.parametrize('identifier', [
        'colour',
        'contrast',
        'eidolonI',
        'eidolonII',
        'eidolonIII',
        'false-colour',
        'high-pass',
        'low-pass',
        'phase-scrambling',
        'power-equalisation',
        'rotation',
        'stylized',
        'sketch',
        'uniform-noise',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_id_long',
        'image_number',
        'experiment_code',
        'condition',
        'truth',
        'category_ground_truth',
        'random_number',
    ])
    def test_fields_present(self, identifier, field):
        stimulus_set = load_stimulus_set(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)

    # tests assembly coords for the 2 "abnormal" sets:
    @pytest.mark.parametrize('identifier', [
        'edge',
        'silhouette',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'image_category',
        'truth',
        'image_variation',
        'condition',
    ])
    def test_fields_present2(self, identifier, field):
        stimulus_set = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)

    # test assembly fields for cue-conflict's odd stimulus_set:
    @pytest.mark.parametrize('identifier', [
        'cue-conflict',
    ])
    @pytest.mark.parametrize('field', [
        'image_id',
        'original_image',
        'truth',
        'category',
        'conflict_image',
        'original_image_category',
        'original_image_variation',
        'conflict_image_category',
        'conflict_image_variation',
        'condition',
    ])
    def test_fields_present3(self, identifier, field):
        stimulus_set = load_dataset(f"brendel.Geirhos2021_{identifier}")
        assert hasattr(stimulus_set, field)


class TestBehavioral:
    def test_count(self):
        assert len(DATASETS) == 12 + 5

    @pytest.mark.parametrize('dataset', DATASETS)
    def test_in_registry(self, dataset):
        identifier = f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"
        assert identifier in benchmark_registry

    def test_mean_ceiling(self):
        benchmarks = [f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency" for dataset in DATASETS]
        benchmarks = [benchmark_registry[benchmark] for benchmark in benchmarks]
        ceilings = [benchmark.ceiling.sel(aggregation='center') for benchmark in benchmarks]
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
        benchmark = benchmark_registry[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center').values.item() == expected_ceiling

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
        benchmark = benchmark_registry[f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        # these features were packaged with condition as int/float. Current xarray versions have trouble when
        # selecting for a float coordinate however, so we had to change the type to string.
        precomputed_features = cast_coordinate_type(precomputed_features, 'condition', newtype=str)
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
        ('resnet-50-pytorch-8deg', approx(0.10256, abs=0.001)),
    ])
    def test_model_mean(self, model, expected_raw_score):
        scores = []
        for dataset in DATASETS:
            benchmark = benchmark_registry[f"brendel.Geirhos2021{dataset.replace('-', '')}-error_consistency"]
            precomputed_features = Path(__file__).parent / f'{model}-Geirhos2021_{dataset}.nc'
            precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
            # these features were packaged with condition as int/float. Current xarray versions have trouble when
            # selecting for a float coordinate however, so we had to change the type to string.
            precomputed_features = cast_coordinate_type(precomputed_features, 'condition', newtype=str)
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
        benchmark = benchmark_registry[f"brendel.Geirhos2021{dataset.replace('-', '')}-top1"]
        # load features
        precomputed_features = Path(__file__).parent / f'{model}-3deg-Geirhos2021_{dataset}.nc'
        precomputed_features = BehavioralAssembly.from_files(file_path=precomputed_features)
        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=None)
        # score
        score = benchmark(precomputed_features)
        assert score.sel(aggregation='center') == expected_accuracy
