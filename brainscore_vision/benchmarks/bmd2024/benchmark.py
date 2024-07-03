import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad

BIBTEX = ""  # to appear in a future article


class _BMD_2024_BehavioralAccuracyDistance(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = load_metric('accuracy_distance')
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        super(_BMD_2024_BehavioralAccuracyDistance, self).__init__(
            identifier=f'BMD_2024_{dataset}Behavioral-AccuracyDistance', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='BMD_2024',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly.stimulus_set['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        labels = candidate.look_at(self._assembly.stimulus_set, number_of_trials=1)
        raw_score = self._metric(labels, target=self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        # ensure score <= 1.0
        if score.values > 1:
            score = Score(np.array(1.))
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def load_assembly(dataset: str) -> BehavioralAssembly:
    assembly = load_dataset(f'BMD_2024_{dataset}')
    return assembly


def BMD2024AccuracyDistance(experiment):
    return _BMD_2024_BehavioralAccuracyDistance(experiment)