import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from brainio.assemblies import BehavioralAssembly, NeuroidAssembly, PropertyAssembly
from brainscore_vision.benchmarks import benchmark_pool, public_benchmark_pool, evaluation_benchmark_pool, \
    engineering_benchmark_pool
from brainscore_vision import benchmark_registry
from brainscore_vision.model_interface import BrainModel
from tests.test_benchmarks import PrecomputedFeatures


# TODO: discuss with Martin


# try to run this and see if tests are run or not.. might have to move all tests to respective directories


class TestPoolList:
    """ ensures that the right benchmarks are in the right benchmark pool """

    def test_contained_global(self, benchmark):
        assert benchmark in benchmark_pool

    def test_contained_public(self, benchmark):
        assert benchmark in public_benchmark_pool


class TestBenchmarkRegistry:
    def test_benchmark_registry(self, benchmark):
        assert benchmark in benchmark_registry


class TestStandardized:

    def test_ceilings(self, benchmark, expected):
        benchmark = benchmark_registry[benchmark]
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == expected

    def test_self_regression(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_registry[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'neuroid')
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10

    def test_self_rdm(self, benchmark, visual_degrees, expected):
        benchmark = benchmark_registry[benchmark]
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score.sel(aggregation='center') == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


class TestPrecomputed:

    def run_test(self, benchmark, file, expected):
        benchmark = benchmark_registry[benchmark]
        precomputed_features = Path(__file__).parent / file
        precomputed_features = NeuroidAssembly.from_files(
            precomputed_features,
            stimulus_set_identifier=benchmark._assembly.stimulus_set.identifier,
            stimulus_set=None)
        precomputed_features = precomputed_features.stack(presentation=['stimulus_path'])
        precomputed_paths = list(map(lambda f: Path(f).name, precomputed_features['stimulus_path'].values))
        # attach stimulus set meta
        stimulus_set = benchmark._assembly.stimulus_set
        expected_stimulus_paths = [stimulus_set.get_stimulus(image_id) for image_id in stimulus_set['stimulus_id']]
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

    def run_test_properties(self, benchmark, files, expected):
        benchmark = benchmark_registry[benchmark]
        from brainscore_vision import get_stimulus_set

        stimulus_identifiers = np.unique(np.array(['dicarlo.Marques2020_blank', 'dicarlo.Marques2020_receptive_field',
                                                   'dicarlo.Marques2020_orientation',
                                                   benchmark._assembly.stimulus_set.identifier]))
        precomputed_features = {}
        for current_stimulus in stimulus_identifiers:
            stimulus_set = get_stimulus_set(current_stimulus)
            path = Path(__file__).parent / files[current_stimulus]
            features = PropertyAssembly.from_files(path,
                                                   stimulus_set_identifier=stimulus_set.identifier,
                                                   stimulus_set=stimulus_set)
            features = features.stack(presentation=['stimulus_path'])
            precomputed_features[current_stimulus] = features
            precomputed_paths = [Path(f).name for f in precomputed_features[current_stimulus]['stimulus_path'].values]
            # attach stimulus set meta
            expected_stimulus_paths = [stimulus_set.get_stimulus(stimulus_id)
                                       for stimulus_id in stimulus_set['stimulus_id']]
            expected_stimulus_paths = list(map(lambda f: Path(f).name, expected_stimulus_paths))
            assert set(precomputed_paths) == set(expected_stimulus_paths)
            for column in stimulus_set.columns:
                precomputed_features[current_stimulus][column] = 'presentation', stimulus_set[column].values

        precomputed_features = PrecomputedFeatures(precomputed_features, visual_degrees=8)
        # score
        score = benchmark(precomputed_features).raw
        assert score.sel(aggregation='center') == expected


class TestVisualDegrees:

    def test_amount_gray(self, benchmark, candidate_degrees, image_id, expected, brainio_home, resultcaching_home,
                         brainscore_home):
        benchmark = benchmark_registry[benchmark]

        class DummyCandidate(BrainModel):
            class StopException(Exception):
                pass

            def visual_degrees(self):
                return candidate_degrees

            def look_at(self, stimuli, number_of_trials=1):
                image = stimuli.get_stimulus(image_id)
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

    def test_repetitions(self, benchmark_identifier):
        """ Tests that benchmarks have repetitions in the stimulus_set """
        benchmark = benchmark_registry[benchmark_identifier]

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
