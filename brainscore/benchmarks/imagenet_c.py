import numpy as np
import xarray as xr

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.imagenet import NUMBER_OF_TRIALS
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel

BIBTEX = """@ARTICLE{Hendrycks2019-di,
   title         = "Benchmarking Neural Network Robustness to Common Corruptions
                    and Perturbations",
   author        = "Hendrycks, Dan and Dietterich, Thomas",
   abstract      = "In this paper we establish rigorous benchmarks for image
                    classifier robustness. Our first benchmark, ImageNet-C,
                    standardizes and expands the corruption robustness topic,
                    while showing which classifiers are preferable in
                    safety-critical applications. Then we propose a new dataset
                    called ImageNet-P which enables researchers to benchmark a
                    classifier's robustness to common perturbations. Unlike
                    recent robustness research, this benchmark evaluates
                    performance on common corruptions and perturbations not
                    worst-case adversarial perturbations. We find that there are
                    negligible changes in relative corruption robustness from
                    AlexNet classifiers to ResNet classifiers. Afterward we
                    discover ways to enhance corruption and perturbation
                    robustness. We even find that a bypassed adversarial defense
                    provides substantial common perturbation robustness.
                    Together our benchmarks may aid future work toward networks
                    that robustly generalize.",
   month         =  mar,
   year          =  2019,
   archivePrefix = "arXiv",
   primaryClass  = "cs.LG",
   eprint        = "1903.12261",
   url           = "https://arxiv.org/abs/1903.12261"
}"""


def Imagenet_C_Noise():
    return Imagenet_C_Category('noise')


def Imagenet_C_Blur():
    return Imagenet_C_Category('blur')


def Imagenet_C_Weather():
    return Imagenet_C_Category('weather')


def Imagenet_C_Digital():
    return Imagenet_C_Category('digital')


class Imagenet_C_Category(BenchmarkBase):
    """
    Runs all ImageNet C benchmarks within a perturbation category, ie: 
    gaussian noise [1-5]
    shot noise [1-5]
    impulse noise [1-5]
    """

    def __init__(self, category):
        category_groups = {
            'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
            'blur': ['glass_blur', 'motion_blur', 'zoom_blur', 'defocus_blur'],
            'weather': ['snow', 'frost', 'fog', 'brightness'],
            'digital': ['pixelate', 'contrast', 'elastic_transform', 'jpeg_compression']
        }
        self._category = category
        self._groups = category_groups[category]
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet_C_Category, self).__init__(identifier='dietterich.Hendrycks2019-top1', version=1,
                                                  ceiling_func=lambda: ceiling,
                                                  parent='ImageNet_C',
                                                  bibtex=BIBTEX)

    def __call__(self, candidate):
        scores = xr.concat([
            Imagenet_C_Group(group)(candidate)
            for group in self._groups
        ], dim='presentation')
        assert len(set(scores['noise_type'].values)) == len(self._groups)
        center = np.mean(scores)
        error = np.std(scores)
        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = scores
        return score


class Imagenet_C_Group(BenchmarkBase):
    """
    Runs a group in imnet C benchmarks, like gaussian noise [1-5]
    """

    def __init__(self, noise_type):
        self._noise_type = noise_type
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet_C_Group, self).__init__(identifier='dietterich.Hendrycks2019-top1', version=1,
                                               ceiling_func=lambda: ceiling,
                                               parent=f'ImageNet_C_{noise_type}',
                                               bibtex=BIBTEX)

    def __call__(self, candidate):
        score = xr.concat([
            Imagenet_C_Individual(f'dietterich.Hendrycks2019.{self._noise_type}_{severity}', self._noise_type)(
                candidate)
            for severity in range(1, 6)
        ], dim='presentation')
        return score


class Imagenet_C_Individual(BenchmarkBase):
    """
    Runs an individual ImageNet C benchmark, like "gaussian_noise_1"
    """

    def __init__(self, benchmark_name, noise_type):
        stimulus_set = brainscore.get_stimulus_set(benchmark_name)
        self._stimulus_set = stimulus_set
        self._similarity_metric = Accuracy()
        self._benchmark_name = benchmark_name
        self._noise_type = noise_type
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet_C_Individual, self).__init__(identifier='dietterich.Hendrycks2019-top1', version=1,
                                                    ceiling_func=lambda: ceiling,
                                                    parent=f'ImageNet_C_{noise_type}',
                                                    bibtex=BIBTEX)

    def __call__(self, candidate):
        # The proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # For now, since almost all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, 'imagenet')
        stimulus_set = self._stimulus_set[list(set(self._stimulus_set.columns) - {'synset'})]  # do not show label
        stimulus_set.identifier = None  # don't pass identifier to disable caching
        predictions = candidate.look_at(stimulus_set, number_of_trials=NUMBER_OF_TRIALS)
        score = self._similarity_metric(
            predictions.sortby('filename'),
            self._stimulus_set.sort_values('filename')['synset'].values
        ).raw

        score = score.assign_coords(
            name=('presentation', [f'{self._benchmark_name}' for _ in range(len(score.presentation))])
        )
        score = score.assign_coords(
            noise_type=('presentation', [f'{self._noise_type}' for _ in range(len(score.presentation))])
        )

        return score
