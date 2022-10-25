import numpy as np

import brainscore
from brainio.assemblies import DataAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.rdm import RDMMetric
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{zhu2019robustness,
            title={Robustness of object recognition under extreme occlusion in humans and computational models},
            author={Zhu, Hongru and Tang, Peng and Park, Jeongho and Park, Soojin and Yuille, Alan},
            journal={arXiv preprint arXiv:1905.04598},
            year={2019}
        }"""

DATASETS = ['extreme_occlusion']

# create functions so that users can import individual benchmarks as e.g. Zhu2019RDM
for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Zhu2019{dataset.replace('-', '')}RDM"
    globals()[identifier] = lambda dataset=dataset: _Zhu2019RDM(dataset)


class _Zhu2019RDM(BenchmarkBase):
    def __init__(self, dataset):
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._fitting_stimuli = brainscore.get_stimulus_set('yuille.Zhu2019_extreme_occlusion')
        self._visual_degrees = 8
        self._number_of_trials = 1

        self._metric = RDMMetric(similarity_function=self._euclidean_similarity, representation_dim='choice')

        super(_Zhu2019RDM, self).__init__(
            identifier=f'yuille.Zhu2019_{dataset}-rdm',
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='yuille.Zhu2019',
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        label_predictions = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        # prepare for RDM metric
        target_assembly = self._human_assembly_categorical_distribution(self._assembly)
        raw_score = self._metric(label_predictions, target_assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

    def _human_assembly_categorical_distribution(self, assembly: DataAssembly) -> DataAssembly:
        # "Image-level representational dissimilarity matrices (RDMs) under extreme occlusion.
        # Each testing image is characterized by a 5-dimensional categorical distribution obtained from [...]
        # human response frequencies for five vehicle categories on that image [...]."
        # We here convert from 19587 trials across 25 subjects to a 500 images x 5 choices assembly
        categories = list(sorted(set(assembly['ground_truth'].values)))

        def categorical(image_responses):
            frequency = np.array([sum(image_responses.values == category) for category in categories])
            frequency = frequency / len(image_responses)
            frequency = type(image_responses)(frequency, coords={'choice': categories}, dims=['choice'])
            return frequency

        stimulus_coords = ['stimulus_id', 'truth', 'filename', 'image_label', 'ground_truth',
                           'occlusion_strength', 'image_number', 'word_image']
        categorical_assembly = assembly.multi_groupby(stimulus_coords).map(categorical)
        return categorical_assembly

    def _euclidean_similarity(self, assembly: DataAssembly) -> DataAssembly:
        # The dissimilarity between two images is measured as the Euclidean distance between two vectors representing
        # the associated categorical distributions."
        # sqrt(sum((a-b)^2))
        distances = np.sqrt(((assembly - assembly.rename(presentation='presentation_right')) ** 2).sum('choice'))
        # Since this is a similarity function, compute `1 - x`.
        # This will get resolved again by the RDM computing 1 - RSA, i.e. `1 - (1 - x) = x`.
        distances = 1 - distances
        return distances


def Zhu2019RDM():
    return _Zhu2019RDM(dataset='extreme_occlusion')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'yuille.Zhu2019_{dataset}')
    return assembly
