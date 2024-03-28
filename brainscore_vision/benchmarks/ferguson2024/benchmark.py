import numpy as np

from brainio.assemblies import walk_coords
from brainscore_vision import load_dataset, load_metric, load_stimulus_set

from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad

BIBTEX = """TBD"""

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']

for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Ferguson2024{dataset}AlignmentMeasure"
    globals()[identifier] = lambda dataset=dataset: _Ferguson2024AlignmentMeasure(dataset)


class _Ferguson2024AlignmentMeasure(BenchmarkBase):
    def __init__(self, dataset):
        self._metric = load_metric('error_consistency')
        self._fitting_stimuli = load_stimulus_set(f'Ferguson2024_{dataset}')
        self._assembly = LazyLoad(lambda: load_dataset(f'Ferguson2024_{dataset}'))
        self._visual_degrees = 8
        self._number_of_trials = 3
        super(_Ferguson2024AlignmentMeasure, self).__init__(
            identifier="Ferguson2024", version=2,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel) -> Score:
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        score = self._metric(probabilities, self._assembly)
        ceiling = self.ceiling
        score = self.ceil_score(score, ceiling)
        return score


def Ferguson2024AlignmentMeasure(experiment):
    return _Ferguson2024AlignmentMeasure(experiment)
