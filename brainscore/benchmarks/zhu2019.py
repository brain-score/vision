import numpy as np
from numpy.random import RandomState
import brainscore
from brainio.assemblies import DataAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from scipy.stats import pearsonr

BIBTEX = """@article{zhu2019robustness,
            title={Robustness of object recognition under extreme occlusion in humans and computational models},
            author={Zhu, Hongru and Tang, Peng and Park, Jeongho and Park, Soojin and Yuille, Alan},
            journal={arXiv preprint arXiv:1905.04598},
            year={2019}
        }"""

DATASETS = ['extreme_occlusion']
NUM_SPLITS = 50


class _Zhu2019Accuracy(BenchmarkBase):

    # Behavioral benchmark: compares model to average humans.
    # human ceiling is calculated by taking NUM_SPLIT split-half reliabilities with *category-level* responses.

    def __init__(self, dataset):
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._fitting_stimuli = brainscore.get_stimulus_set('Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        self._visual_degrees = 8
        self._number_of_trials = 1
        self._metric = Accuracy()
        self._ceiling = SplitHalvesConsistencyZhu(num_splits=NUM_SPLITS, split_coordinate="subject")

        super(_Zhu2019Accuracy, self).__init__(
            identifier=f'Zhu2019_{dataset}-accuracy',
            parent='Zhu2019',
            ceiling_func=lambda: self._ceiling(assembly=self._assembly),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        source = _human_assembly_categorical_distribution(self._assembly, collapse=False)
        target = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(target, source)
        ceiling = self._ceiling(self._assembly)
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


class _Zhu2019Accuracy_Engineering(BenchmarkBase):
    # engineering benchmark: compares model to ground_truth
    def __init__(self, dataset):
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._fitting_stimuli = brainscore.get_stimulus_set('Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: load_assembly(dataset).stimulus_set)
        self._visual_degrees = 8
        self._number_of_trials = 1

        self._metric = Accuracy()

        super(_Zhu2019Accuracy_Engineering, self).__init__(
            identifier=f'Zhu2019_{dataset}-accuracy-engineering',
            parent='Zhu2019',
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        model_results = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        ground_truth = model_results["ground_truth"]

        # compare model with stimulus_set (ground_truth)
        raw_score = self._metric(model_results, ground_truth)
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


"""
Convert from 19587 trials across 25 subjects to a 500 images x 5 choices assembly.
This is needed, as not every subject saw every image, and this allows a cross-category comparison
"""


def _human_assembly_categorical_distribution(assembly: DataAssembly, collapse) -> DataAssembly:
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


# ceiling method:
class SplitHalvesConsistencyZhu:
    def __init__(self, num_splits: int, split_coordinate: str):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate

    def __call__(self, assembly) -> Score:

        consistencies, uncorrected_consistencies = [], []
        splits = range(self.num_splits)
        random_state = np.random.RandomState(0)

        # loops through until (self.num_splits) of splits have occurred
        i = 0
        while i < self.num_splits:
            print(i)
            num_subjects = len(set(assembly["subject"].values))
            half1_subjects = random_state.choice(range(1, num_subjects), (num_subjects // 2), replace=False)
            half1 = assembly[
                {'presentation': [subject in half1_subjects for subject in assembly['subject'].values]}]
            half2 = assembly[
                {'presentation': [subject not in half1_subjects for subject in assembly['subject'].values]}]
            categorical_assembly1 = _human_assembly_categorical_distribution(half1, collapse=True)
            categorical_assembly2 = _human_assembly_categorical_distribution(half2, collapse=True)

            # the two categorical assemblies should be 5x5. however, during a split, subjects might not see all images
            # the others do. In which case, we pass.
            try:
                consistency = pearsonr(categorical_assembly1.values.flatten(), categorical_assembly2.values.flatten())[
                    0]
                uncorrected_consistencies.append(consistency)
                # Spearman-Brown correction for sub-sampling
                corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
                consistencies.append(corrected_consistency)
                i += 1
            except ValueError:
                pass
        consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
        uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
        average_consistency = consistencies.median('split')
        average_consistency.attrs['raw'] = consistencies
        average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
        return average_consistency


def Zhu2019Accuracy():
    return _Zhu2019Accuracy(dataset='extreme_occlusion')


def Zhu2019Accuracy_Engineering():
    return _Zhu2019Accuracy_Engineering(dataset='extreme_occlusion')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'Zhu2019_{dataset}')
    return assembly
