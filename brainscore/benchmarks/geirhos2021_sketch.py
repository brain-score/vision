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


class _Geirhos2021Sketch(BenchmarkBase):
    def __init__(self, metric, metric_identifier):
        self._metric = metric
        # TODO: fix stimulus set for fitting, should be different from test images
        self._fitting_stimuli = brainscore.get_stimulus_set('brendel.Geirhos2021_sketch')
        # TODO: subject should not be part of stimulus set, images are independent of subjects
        # TODO: rename category_ground_truth to two columns: category and ground truth
        self._fitting_stimuli['image_label'] = self._fitting_stimuli['category_ground_truth']  # required by metric
        self._assembly = LazyLoad(lambda: load_assembly())
        self._visual_degrees = 3

        # TODO
        self._number_of_trials = 1

        super(_Geirhos2021Sketch, self).__init__(
            identifier='brendel.Geirhos2021_sketch' + metric_identifier, version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
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

    def ceil_score(self, score, ceiling):
        assert set(score.raw['split'].values) == set(ceiling.raw['split'].values)
        split_scores = []
        for split in ceiling.raw['split'].values:
            split_score = score.raw.sel(split=split)
            split_ceiling = ceiling.raw.sel(split=split)
            ceiled_split_score = split_score / np.sqrt(split_ceiling)
            ceiled_split_score = ceiled_split_score.expand_dims('split')
            ceiled_split_score['split'] = [split]
            split_scores.append(ceiled_split_score)
        split_scores = Score.merge(*split_scores)
        split_scores = apply_aggregate(self._metric.aggregate, split_scores)
        split_scores.attrs[Score.RAW_VALUES_KEY] = score  # this will override raw per-split ceiled scores which is ok
        split_scores.attrs['ceiling'] = ceiling
        return split_scores


def Geirhos2021SketchI1():
    return _Geirhos2021Sketch(metric=I1(), metric_identifier='i1')


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
