import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.image_level_behavior import I2n
from brainscore.metrics.transformations import apply_aggregate
from brainscore.model_interface import BrainModel
from brainscore.benchmarks.screen import place_on_screen
from brainscore.utils import LazyLoad


class DicarloRajalingham2018I2n(BenchmarkBase):
    def __init__(self):
        self._metric = I2n()
        self._fitting_stimuli = brainscore.get_stimulus_set('dicarlo.objectome.public')
        self._assembly = LazyLoad(lambda: load_assembly('private'))
        self._visual_degrees = 8
        super(DicarloRajalingham2018I2n, self).__init__(
            identifier='dicarlo.Rajalingham2018-i2n', parent='behavior',version=2,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            bibtex="""@article {Rajalingham7255,
                        author = {Rajalingham, Rishi and Issa, Elias B. and Bashivan, Pouya and Kar, Kohitij and Schmidt, Kailyn and DiCarlo, James J.},
                        title = {Large-Scale, High-Resolution Comparison of the Core Visual Object Recognition Behavior of Humans, Monkeys, and State-of-the-Art Deep Artificial Neural Networks},
                        volume = {38},
                        number = {33},
                        pages = {7255--7269},
                        year = {2018},
                        doi = {10.1523/JNEUROSCI.0388-18.2018},
                        publisher = {Society for Neuroscience},
                        abstract = {Primates, including humans, can typically recognize objects in visual images at a glance despite naturally occurring identity-preserving image transformations (e.g., changes in viewpoint). A primary neuroscience goal is to uncover neuron-level mechanistic models that quantitatively explain this behavior by predicting primate performance for each and every image. Here, we applied this stringent behavioral prediction test to the leading mechanistic models of primate vision (specifically, deep, convolutional, artificial neural networks; ANNs) by directly comparing their behavioral signatures against those of humans and rhesus macaque monkeys. Using high-throughput data collection systems for human and monkey psychophysics, we collected more than one million behavioral trials from 1472 anonymous humans and five male macaque monkeys for 2400 images over 276 binary object discrimination tasks. Consistent with previous work, we observed that state-of-the-art deep, feedforward convolutional ANNs trained for visual categorization (termed DCNNIC models) accurately predicted primate patterns of object-level confusion. However, when we examined behavioral performance for individual images within each object discrimination task, we found that all tested DCNNIC models were significantly nonpredictive of primate performance and that this prediction failure was not accounted for by simple image attributes nor rescued by simple model modifications. These results show that current DCNNIC models cannot account for the image-level behavioral patterns of primates and that new ANN models are needed to more precisely capture the neural mechanisms underlying primate object vision. To this end, large-scale, high-resolution primate behavioral benchmarks such as those obtained here could serve as direct guides for discovering such models.SIGNIFICANCE STATEMENT Recently, specific feedforward deep convolutional artificial neural networks (ANNs) models have dramatically advanced our quantitative understanding of the neural mechanisms underlying primate core object recognition. In this work, we tested the limits of those ANNs by systematically comparing the behavioral responses of these models with the behavioral responses of humans and monkeys at the resolution of individual images. Using these high-resolution metrics, we found that all tested ANN models significantly diverged from primate behavior. Going forward, these high-resolution, large-scale primate behavioral benchmarks could serve as direct guides for discovering better ANN models of the primate visual system.},
                        issn = {0270-6474},
                        URL = {https://www.jneurosci.org/content/38/33/7255},
                        eprint = {https://www.jneurosci.org/content/38/33/7255.full.pdf},
                        journal = {Journal of Neuroscience}
                    }""")

    def __call__(self, candidate: BrainModel):
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(stimulus_set)
        score = self._metric(probabilities, self._assembly)
        score = self.ceil_score(score, self.ceiling)
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


def load_assembly(access='private'):
    assembly = brainscore.get_assembly(f'dicarlo.Rajalingham2018.{access}')
    assembly['correct'] = assembly['choice'] == assembly['sample_obj']
    return assembly
