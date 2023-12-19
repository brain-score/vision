import numpy as np

from brainscore_core import Score
from brainscore_vision import load_metric, load_stimulus_set, load_dataset
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metric_helpers.transformations import apply_aggregate
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad

BIBTEX = """@article {Rajalingham240614,
                author = {Rajalingham, Rishi and Issa, Elias B. and Bashivan, Pouya and Kar, Kohitij and Schmidt, Kailyn and DiCarlo, James J.},
                title = {Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks},
                elocation-id = {240614},
                year = {2018},
                doi = {10.1101/240614},
                publisher = {Cold Spring Harbor Laboratory},
                abstract = {Primates{\textemdash}including humans{\textemdash}can typically recognize objects in visual images at a glance even in the face of naturally occurring identity-preserving image transformations (e.g. changes in viewpoint). A primary neuroscience goal is to uncover neuron-level mechanistic models that quantitatively explain this behavior by predicting primate performance for each and every image. Here, we applied this stringent behavioral prediction test to the leading mechanistic models of primate vision (specifically, deep, convolutional, artificial neural networks; ANNs) by directly comparing their behavioral signatures against those of humans and rhesus macaque monkeys. Using high-throughput data collection systems for human and monkey psychophysics, we collected over one million behavioral trials for 2400 images over 276 binary object discrimination tasks. Consistent with previous work, we observed that state-of-the-art deep, feed-forward convolutional ANNs trained for visual categorization (termed DCNNIC models) accurately predicted primate patterns of object-level confusion. However, when we examined behavioral performance for individual images within each object discrimination task, we found that all tested DCNNIC models were significantly non-predictive of primate performance, and that this prediction failure was not accounted for by simple image attributes, nor rescued by simple model modifications. These results show that current DCNNIC models cannot account for the image-level behavioral patterns of primates, and that new ANN models are needed to more precisely capture the neural mechanisms underlying primate object vision. To this end, large-scale, high-resolution primate behavioral benchmarks{\textemdash}such as those obtained here{\textemdash}could serve as direct guides for discovering such models.SIGNIFICANCE STATEMENT Recently, specific feed-forward deep convolutional artificial neural networks (ANNs) models have dramatically advanced our quantitative understanding of the neural mechanisms underlying primate core object recognition. In this work, we tested the limits of those ANNs by systematically comparing the behavioral responses of these models with the behavioral responses of humans and monkeys, at the resolution of individual images. Using these high-resolution metrics, we found that all tested ANN models significantly diverged from primate behavior. Going forward, these high-resolution, large-scale primate behavioral benchmarks could serve as direct guides for discovering better ANN models of the primate visual system.},
                URL = {https://www.biorxiv.org/content/early/2018/02/12/240614},
                eprint = {https://www.biorxiv.org/content/early/2018/02/12/240614.full.pdf},
                journal = {bioRxiv}
            }"""


class _DicarloRajalingham2018(BenchmarkBase):
    def __init__(self, metric, metric_identifier):
        self._metric = metric
        self._fitting_stimuli = load_stimulus_set('dicarlo.objectome.public')
        self._assembly = LazyLoad(lambda: load_assembly('private'))
        self._visual_degrees = 8
        self._number_of_trials = 2
        super(_DicarloRajalingham2018, self).__init__(
            identifier='dicarlo.Rajalingham2018-' + metric_identifier, version=2,
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


def DicarloRajalingham2018I2n():
    return _DicarloRajalingham2018(metric=load_metric('i2n'), metric_identifier='i2n')


def load_assembly(access='private'):
    assembly = load_dataset(f'Rajalingham2018.{access}')
    assembly['correct'] = assembly['choice'] == assembly['sample_obj']
    return assembly
