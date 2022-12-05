import numpy as np
from numpy import ndarray
from numpy.random import RandomState

import brainscore
from brainio.assemblies import DataAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen

from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from brainscore.metrics.error_consistency import ErrorConsistency

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
    identifier = f"Zhu2019{dataset.replace('-', '')}Accuracy"
    globals()[identifier] = lambda dataset=dataset: _Zhu2019Accuracy(dataset)

    # engineering benchmark
    identifier = f"Zhu2019{dataset.replace('-', '')}AccuracyEngineering"
    globals()[identifier] = lambda dataset=dataset: _Zhu2019Accuracy_Engineering(dataset)


class _Zhu2019Accuracy(BenchmarkBase):
    # behavioral benchmark: compares model: average humans
    def __init__(self, dataset):
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._fitting_stimuli = brainscore.get_stimulus_set('yuille.Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        self._visual_degrees = 8
        self._number_of_trials = 1

        self._metric = Accuracy()

        super(_Zhu2019Accuracy, self).__init__(
            identifier=f'yuille.Zhu2019_{dataset}-accuracy',
            parent='yuille.Zhu2019',
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        human_results = _human_assembly_categorical_distribution(self._assembly, collapse=False)
        model_results = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        # compare model to human
        raw_score = self._metric(model_results, human_results)

        ceiling = calculate_ceiling(self._assembly, self._metric)
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


class _Zhu2019Accuracy_Engineering(BenchmarkBase):
    # engineering benchmark: compares model to ground_truth
    def __init__(self, dataset):
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._fitting_stimuli = brainscore.get_stimulus_set('yuille.Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        self._visual_degrees = 8
        self._number_of_trials = 1

        self._metric = Accuracy()

        super(_Zhu2019Accuracy_Engineering, self).__init__(
            identifier=f'yuille.Zhu2019_{dataset}-accuracy-engineering',
            parent='yuille.Zhu2019',
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        model_results = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        ground_truth = stimulus_set.sort_values("stimulus_id")["ground_truth"].values

        # compare model with stimulus_set (ground_truth)
        raw_score = self._metric(model_results, ground_truth)

        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def _human_assembly_categorical_distribution(assembly: DataAssembly, collapse) -> DataAssembly:
    # We here convert from 19587 trials across 25 subjects to a 500 images x 5 choices assembly
    categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]

    def categorical(image_responses):
        frequency = np.array([sum(image_responses.values == category) for category in categories])
        frequency = frequency / len(image_responses)
        frequency = type(image_responses)(frequency, coords={'choice': categories}, dims=['choice'])
        return frequency

    stimulus_coords = ['stimulus_id', 'truth', 'filename', 'image_label', 'ground_truth',
                       'occlusion_strength', 'image_number', 'word_image']
    categorical_assembly = assembly.multi_groupby(stimulus_coords).map(categorical)
    labels = get_choices(categorical_assembly, categories=categories)

    if collapse:
        categorical_assembly = categorical_assembly.groupby('image_label').sum('presentation')
        return categorical_assembly

    return labels


# takes 5-way response vector and returns category of highest response
def get_choices(predictions, categories):
    choices = []
    for prediction in predictions:
        choice_index = list(prediction.values).index(max(prediction))
        choice = categories[choice_index]
        choices.append(choice)
    return np.array(choices)


def calculate_ceiling(assembly, metric):
    import random
    from scipy.stats import pearsonr
    ceilings = []
    while len(ceilings) < 10:
        half1_subjects = random.sample(range(1, 26), 12)
        half1 = assembly[
            {'presentation': [subject in half1_subjects for subject in assembly['subject'].values]}]
        half2 = assembly[
            {'presentation': [subject not in half1_subjects for subject in assembly['subject'].values]}]

        categorical_assembly1 = _human_assembly_categorical_distribution(half1, collapse=True)
        categorical_assembly2 = _human_assembly_categorical_distribution(half2, collapse=True)

        try:
            ceiling = pearsonr(categorical_assembly1.values.flatten(), categorical_assembly2.values.flatten())[0]
            ceilings.append(ceiling)
        except ValueError:
            pass

    return np.mean(ceilings)

def Zhu2019Accuracy():
    return _Zhu2019Accuracy(dataset='extreme_occlusion')


def Zhu2019Accuracy_Engineering():
    return _Zhu2019Accuracy_Engineering(dataset='extreme_occlusion')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'yuille.Zhu2019_{dataset}')
    return assembly
