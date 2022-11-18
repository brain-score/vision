import numpy as np
from numpy.random import RandomState

import brainscore
from brainio.assemblies import DataAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from scipy.stats import pearsonr
import random
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
        target_assembly = self._human_assembly_categorical_distribution(self._assembly, collapse=True)
        label_predictions = self._collapse_categories(label_predictions, show_rdm=True)
        # raw_score = self._metric(label_predictions, target_assembly)
        raw_score = pearsonr(target_assembly.values.flatten(), label_predictions.values.flatten())[0]

        # calculate intersubject split-half reliability ceiling, averaged over 10 split halves:
        ceilings = []
        while len(ceilings) < 10:
            half1_subjects = random.sample(range(1, 26), 12)
            half1 = self._assembly[{'presentation': [subject in half1_subjects for subject in self._assembly['subject'].values]}]
            half2 = self._assembly[{'presentation': [subject not in half1_subjects for subject in self._assembly['subject'].values]}]

            categorical_assembly1 = self._human_assembly_categorical_distribution(half1, collapse=True)
            categorical_assembly2 = self._human_assembly_categorical_distribution(half2, collapse=True)

            # sometimes the random split forces the resultant categorical assembly to be 4x4,
            # not 5x5. If this is the case, just pass and re-split, until 10 ceilings are calculated.
            try:
                ceiling = pearsonr(categorical_assembly1.values.flatten(), categorical_assembly2.values.flatten())[0]
                ceilings.append(ceiling)
            except ValueError:
                pass

        ceiling = np.mean(ceilings)
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

    def _human_assembly_categorical_distribution(self, assembly: DataAssembly, collapse) -> DataAssembly:
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

        if collapse:
            categorical_assembly = self._collapse_categories(categorical_assembly)

        return categorical_assembly

    def _euclidean_similarity(self, assembly: DataAssembly) -> DataAssembly:
        # The dissimilarity between two images is measured as the Euclidean distance between two vectors representing
        # the associated categorical distributions."
        # sqrt(sum((a-b)^2))
        distances = np.sqrt(((assembly - assembly.rename(image_label='presentation_right')) ** 2).sum('choice'))
        # Since this is a similarity function, compute `1 - x`.
        # This will get resolved again by the RDM computing 1 - RSA, i.e. `1 - (1 - x) = x`.
        distances = 1 - distances
        return distances

    def _collapse_categories(self, rdm_assembly, show_rdm=False):
        categorical_assembly = rdm_assembly.groupby('image_label').sum('presentation')

        # RDM Plots:
        categories = ["aeroplane", "bicycle", "bus", "car", "motorbike"]
        target_array = np.array(categorical_assembly).T
        import seaborn as sn
        import matplotlib.pyplot as plt
        plt.clf()
        hm = sn.heatmap(data=target_array, xticklabels=categories, yticklabels=categories)

        if show_rdm:
            plt.show()
        return categorical_assembly


def Zhu2019RDM():
    return _Zhu2019RDM(dataset='extreme_occlusion')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'yuille.Zhu2019_{dataset}')
    return assembly
