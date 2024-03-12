import brainscore_vision
from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad

BIBTEX = ""  # to appear in a future article


class _Scialom2024BehavioralErrorConsistency(BenchmarkBase):
    def __init__(self, dataset):
        self._metric = load_metric('error_consistency')
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        self._visual_degrees = 8
        self._number_of_trials = 1

        super(_Scialom2024BehavioralErrorConsistency, self).__init__(
            identifier=f'Scialom2024{dataset}-error_consistency', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='Scialom2024',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


class _Scialom2024BehavioralAccuracyDistance(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = load_metric('accuracy_distance')
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        super(_Scialom2024BehavioralAccuracyDistance, self).__init__(
            identifier=f'Scialom2024{dataset}-behavioral_accuracy', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='Scialom2024-top1',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._stimulus_set['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        labels = candidate.look_at(self._stimulus_set, number_of_trials=1)
        score = self._metric(labels, target=self._stimulus_set['truth'].values)
        return score


class _Scialom2024EngineeringAccuracy(BenchmarkBase):
    # engineering/ML benchmark
    def __init__(self, dataset):
        self._metric = load_metric('accuracy')
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        super(_Scialom2024EngineeringAccuracy, self).__init__(
            identifier=f'Scialom2024{dataset}-engineering_accuracy', version=1,
            ceiling_func=lambda: Score(1),
            parent='Scialom2024-top1',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._stimulus_set['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        labels = candidate.look_at(self._stimulus_set, number_of_trials=10)
        score = self._metric(labels, target=self._stimulus_set['truth'].values)
        return score


def load_assembly(dataset: str) -> BehavioralAssembly:
    assembly = brainscore_vision.load_dataset(f'Scialom2024{dataset}')
    return assembly
