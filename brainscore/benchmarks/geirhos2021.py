import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.image_level_behavior import I1
from brainscore.metrics.transformations import apply_aggregate
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{geirhos2021partial,
              title={Partial success in closing the gap between human and machine vision},
              author={Geirhos, Robert and Narayanappa, Kantharaju and Mitzkus, Benjamin and Thieringer, Tizian and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
              journal={Advances in Neural Information Processing Systems},
              volume={34},
              year={2021}
        }"""


class Geirhos2021Sketch(BenchmarkBase):
    def __init__(self):
        self._metric = CohensKappa()
        # TODO: fix stimulus set for fitting, should be different from test images
        self._fitting_stimuli = brainscore.get_stimulus_set('brendel.Geirhos2021_sketch')
        # TODO: subject should not be part of stimulus set, images are independent of subjects
        # TODO: rename category_ground_truth to two columns: category and ground truth
        self._fitting_stimuli['image_label'] = self._fitting_stimuli['category_ground_truth']  # required by metric
        self._assembly = LazyLoad(lambda: load_assembly())
        self._visual_degrees = 3

        # TODO
        self._number_of_trials = 1

        super(Geirhos2021Sketch, self).__init__(
            identifier='brendel.Geirhos2021_sketch-cohen_kappa', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.probabilities, choice_labels)  # TODO
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        score = self._metric(probabilities, self._assembly)
        ceiling = self.ceiling
        score = self.ceil_score(score, ceiling)
        return score


def load_assembly():
    assembly = brainscore.get_assembly('brendel.Geirhos2021_sketch')

    # add needed fields to assembly:
    assembly['choice'] = ('presentation', assembly.values)
    assembly['truth'] = assembly['category']
    assembly['sample_obj'] = assembly['category']
    assembly['correct'] = assembly['choice'] == assembly['truth']
    # assembly['dist_obj'] = assembly['correct']

    # drop the 40 rows with "na" as subject response -> cannot use for correlations, etc.
    assembly_processed = assembly.where(assembly.choice != "na", drop=True)

    return assembly_processed
