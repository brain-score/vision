from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from brainio.assemblies import NeuroidAssembly, PropertyAssembly
from brainscore_vision import load_benchmark
from brainscore_vision.model_interface import BrainModel
from . import PrecomputedFeatures


class TestStandardized:
    def ceilings_test(self, benchmark: str, expected: float):
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected

    def self_regression_test(self, benchmark: str, visual_degrees: int, expected: float):
        benchmark = load_benchmark(benchmark)
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'neuroid')
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10

    def self_rdm_test(self, benchmark: str, visual_degrees: int, expected: float):
        benchmark = load_benchmark(benchmark)
        score = benchmark(PrecomputedFeatures(benchmark._assembly, visual_degrees=visual_degrees)).raw
        assert score == expected
        raw_values = score.attrs['raw']
        assert hasattr(raw_values, 'split')
        assert len(raw_values['split']) == 10


class TestPrecomputed:
    def run_test(self, benchmark: str, precomputed_features_filepath: str, expected: float):
        benchmark = load_benchmark(benchmark)
        precomputed_features = NeuroidAssembly.from_files(
            precomputed_features_filepath,
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
        assert score == expected

    def run_test_properties(self, benchmark: str, files: dict, expected: float):
        benchmark = load_benchmark(benchmark)
        from brainscore_vision import load_stimulus_set

        stimulus_identifiers = np.unique(np.array(['dicarlo.Marques2020_blank', 'dicarlo.Marques2020_receptive_field',
                                                   'dicarlo.Marques2020_orientation',
                                                   benchmark._assembly.stimulus_set.identifier]))
        precomputed_features = {}
        for current_stimulus in stimulus_identifiers:
            stimulus_set = load_stimulus_set(current_stimulus)
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
        assert score == expected


class TestVisualDegrees:
    def amount_gray_test(self, benchmark: str, candidate_degrees: int, image_id: str, expected: float,
                         brainio_home: Path, resultcaching_home: Path, brainscore_home: Path):
        benchmark = load_benchmark(benchmark)

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
    def repetitions_test(self, benchmark_identifier: str):
        """ Tests that benchmarks have repetitions in the stimulus_set """
        benchmark = load_benchmark(benchmark_identifier)

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
