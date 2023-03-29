import numpy as np
from scipy.stats import pearsonr

import brainscore
from brainio.assemblies import DataAssembly
from brainio.assemblies import walk_coords, BehavioralAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.response_match import ResponseMatch
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from brainscore.metrics.ceiling import SpearmanBrownCorrection
from tqdm import tqdm
from numpy.random import RandomState

BIBTEX = """@article{zhu2019robustness,
            title={Robustness of object recognition under extreme occlusion in humans and computational models},
            author={Zhu, Hongru and Tang, Peng and Park, Jeongho and Park, Soojin and Yuille, Alan},
            journal={arXiv preprint arXiv:1905.04598},
            year={2019}
        }"""

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 1
NUM_SPLITS = 50


class _Zhu2019ResponseMatch(BenchmarkBase):
    """
    Behavioral benchmark: compares model to average humans.
    human ceiling is calculated by taking `NUM_SPLIT` split-half reliabilities with *category-level* responses.
    This response_match benchmark compares the top average human response (out of 5 categories) per image to
    the model response per image.
    """

    def __init__(self):
        self._assembly = LazyLoad(lambda: brainscore.get_assembly('Zhu2019_extreme_occlusion'))
        self._ceiler = HalvesZhu(metric=ResponseMatch(), split_coordinate="subject", num_splits=25)
        self._fitting_stimuli = brainscore.get_stimulus_set('Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: self._assembly.stimulus_set)
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = NUMBER_OF_TRIALS
        self._metric = ResponseMatch()

        super(_Zhu2019ResponseMatch, self).__init__(
            identifier='Zhu2019_extreme_occlusion-response_match',
            parent='Ferguson2023',
            ceiling_func=lambda: self._ceiler(assembly=self._assembly),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        human_responses = _human_assembly_categorical_distribution(self._assembly, collapse=False)
        predictions = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        predictions = predictions.sortby("stimulus_id")

        # ensure alignment of data:
        assert set(predictions["stimulus_id"].values == human_responses["stimulus_id"].values) == {True}

        raw_score = self._metric(predictions, human_responses)
        ceiling = self._ceiling(self._assembly)
        score = raw_score / self.ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        return score


class _Zhu2019Accuracy(BenchmarkBase):
    """ engineering benchmark: compares model to ground truth image labels """

    def __init__(self):
        self._fitting_stimuli = brainscore.get_stimulus_set('Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: brainscore.get_assembly(f'Zhu2019_extreme_occlusion').stimulus_set)
        self._assembly = LazyLoad(lambda: brainscore.get_assembly(f'Zhu2019_extreme_occlusion'))
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = NUMBER_OF_TRIALS

        self._metric = Accuracy()

        super(_Zhu2019Accuracy, self).__init__(
            identifier='Zhu2019_extreme_occlusion-accuracy',
            parent='Zhu2019',
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):

        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))

        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        predictions = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        predictions = predictions.sortby("stimulus_id")

        # grab ground_truth from predictions (linked via stimulus_id) instead of stimulus set, to ensure sort
        ground_truth = predictions["ground_truth"].sortby("stimulus_id")

        # compare model with stimulus_set (ground_truth)
        raw_score = self._metric(predictions, ground_truth)
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')

        # cap score at 1 if ceiled score > 1
        if score[(score['aggregation'] == 'center')] > 1:
            score.__setitem__({'aggregation': score['aggregation'] == 'center'}, 1)

        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def _human_assembly_categorical_distribution(assembly: LazyLoad, collapse: bool) -> DataAssembly:
    """
    Convert from 19587 trials across 25 subjects to a 500 images x 5 choices assembly.
    This is needed, as not every subject saw every image, and this allows a cross-category comparison
    """

    choice_labels = set(assembly['truth'].values)
    categories = list(sorted(choice_labels))

    def categorical(image_responses):
        frequency = np.array([sum(image_responses.values == category) for category in categories])
        frequency = frequency / len(image_responses)
        frequency = type(image_responses)(frequency, coords={'choice': categories}, dims=['choice'])
        return frequency

    stimulus_coords = ['stimulus_id', 'truth', 'filename', 'image_label', 'ground_truth',
                       'occlusion_strength', 'image_number', 'word_image']
    categorical_assembly = assembly.multi_groupby(stimulus_coords).map(categorical)

    # get top average human response, across categories per image.
    labels = get_choices(categorical_assembly, categories=categories)

    if collapse:
        categorical_assembly = categorical_assembly.groupby('image_label').sum('presentation')
        return categorical_assembly

    return labels


# takes 5-way softmax vector and returns category of highest response
def get_choices(predictions:BehavioralAssembly, categories:list) -> BehavioralAssembly:

    # make sure predictions (choices) are aligned with categories:
    assert set(predictions["choice"].values == categories) == {True}

    indexes = list(range(0, 5))
    mapping = dict(zip(indexes, categories))
    extra_coords = {}
    prediction_indices = predictions.values.argmax(axis=1)
    choices = [mapping[index] for index in prediction_indices]
    extra_coords['computed_choice'] = ('presentation', choices)
    coords = {**{coord: (dims, values) for coord, dims, values in walk_coords(predictions['presentation'])},
              **{'label': ('presentation', choices)},
              **extra_coords}
    final = BehavioralAssembly([choices], coords=coords, dims=['choice', 'presentation'])
    return final.sortby("stimulus_id")


class HalvesZhu:
    def __init__(self, metric, split_coordinate: str, num_splits: int = 10):
        """
        :param metric: compare assembly splits
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param num_splits: how many splits to estimate the consistency over
        """
        self.metric = metric
        self.split_coordinate = split_coordinate
        self.num_splits = num_splits

    def __call__(self, assembly) -> Score:
        consistencies = []
        images = set(assembly['stimulus_id'].values)

        for image in images:
            print(image)
            image_consistencies = []
            single_category_assembly = assembly[
                {'presentation': [_image == image for _image in assembly['stimulus_id'].values]}]

            # split the images in half randomly to pass to metric
            for i in range(self.num_splits):
                random_state = np.random.RandomState(i)
                print(i)
                trials = single_category_assembly["choice"].values
                random_state.shuffle(trials)
                trials = trials.tolist()
                if len(trials) % 2 != 0:
                    trials.pop(0)
                half_1 = trials[:len(trials) // 2]
                half_2 = trials[len(trials) // 2:]
                consistency = self.metric(BehavioralAssembly(half_1), BehavioralAssembly(half_2))
                image_consistencies.append(consistency[(consistency['aggregation'] == 'center')])

            average_consistency = np.mean(image_consistencies)
            consistencies.append(average_consistency)
        avg = np.mean(consistencies)
        return avg

def Zhu2019ResponseMatch():
    return _Zhu2019ResponseMatch()


def Zhu2019Accuracy():
    return _Zhu2019Accuracy()
