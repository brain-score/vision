"""
This data was collected by Kohitij Kar on Amazon mechanical turk.
It was first used in Kubilius*, Schrimpf*, et al. NeurIPS (2019)
https://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns.
"""
import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.image_level_behavior import I2n
from brainscore.metrics.transformations import apply_aggregate
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = None


class _DicarloKar2018(BenchmarkBase):
    def __init__(self, metric, metric_identifier):
        self._metric = metric
        # self._fitting_stimuli = brainscore.get_stimulus_set('dicarlo.Kar2018coco_color.public')
        # FIXME
        from packaging.dicarlo.kar2018_cocobehavior import main as load_assemblies
        _, _, public_stimuli, private_stimuli = load_assemblies()
        self._fitting_stimuli = public_stimuli
        self._assembly = LazyLoad(lambda: load_assembly('private'))
        self._visual_degrees = 8
        self._number_of_trials = 2
        super(_DicarloKar2018, self).__init__(
            identifier='dicarlo.Kar2018-' + metric_identifier, version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly, skipna=True),
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


def DicarloKar2018I2n():
    return _DicarloKar2018(metric=I2n(), metric_identifier='i2n')


def load_assembly(access='private'):
    # FIXME
    from packaging.dicarlo.kar2018_cocobehavior import main as load_assemblies
    public, private, public_stimuli, private_stimuli = load_assemblies()
    public.attrs['stimulus_set'], private.attrs['stimulus_set'] = public_stimuli, private_stimuli
    assemblies = {'public': public, 'private': private}
    assembly = assemblies[access]
    # assembly = brainscore.get_assembly(f'dicarlo.Kar2018coco_behavior.{access}')
    assembly['correct'] = assembly['choice'] == assembly['sample_obj']
    return assembly
