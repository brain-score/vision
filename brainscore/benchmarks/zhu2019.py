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
        self._ceiler = OneVsManyZhu(split_coordinate="subject")
        self._fitting_stimuli = brainscore.get_stimulus_set('Zhu2019_extreme_occlusion')
        self._stimulus_set = LazyLoad(lambda: self._assembly.stimulus_set)
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = NUMBER_OF_TRIALS
        self._metric = ResponseMatch()

        super(_Zhu2019ResponseMatch, self).__init__(
            identifier='Zhu2019_extreme_occlusion-response_match',
            parent='Zhu2019',
            ceiling_func=lambda: self._ceiler(assembly=self._assembly),
            bibtex=BIBTEX, version=1)

    def __call__(self, candidate: BrainModel):
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        human_responses = _human_assembly_categorical_distribution(self._assembly, collapse=False)
        predictions = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        predictions = predictions.sortby("stimulus_id")
        raw_score = self._metric(predictions, human_responses)
        ceiling = self._ceiler(self._assembly)
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
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
        # hard code categories, as sorting alphabetically leads to mismatched IDs
        categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]
        candidate.start_task(BrainModel.Task.label, categories)
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


def _human_assembly_categorical_distribution(assembly: DataAssembly, collapse) -> DataAssembly:
    """
    Convert from 19587 trials across 25 subjects to a 500 images x 5 choices assembly.
    This is needed, as not every subject saw every image, and this allows a cross-category comparison
    """

    categories = ["car", "aeroplane", "motorbike", "bicycle", "bus"]

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
def get_choices(predictions, categories):
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


# ceiling method:
class OneVsManyZhu:
    def __init__(self, split_coordinate: str):
        """
        :param split_coordinate: over which coordinate to split the assembly into halves
        """
        self.split_coordinate = split_coordinate

    def __call__(self, assembly) -> Score:
        consistencies, uncorrected_consistencies = [], []
        splits = range(len(set(assembly["subject"].values)))

        # For each subject, compare that subject's responses to the pool of other subjects that also saw
        # that category. 
        for subject in set(assembly["subject"].values):
            single_subject_assembly = assembly[
                {'presentation': [_subject == subject for _subject in assembly['subject'].values]}]
            subject_seen_categories = set(single_subject_assembly["image_label"].values)
            pool_assembly = assembly[
                {'presentation': [_subject != subject for _subject in assembly['subject'].values]}]
            # filter to only those stimuli that the held-out subject has also seen
            pool_assembly = pool_assembly[pool_assembly["image_label"] in subject_seen_categories]
            # compute categoricals and compare
            single_categorical = _human_assembly_categorical_distribution(single_subject_assembly, collapse=True)
            pool_categorical = _human_assembly_categorical_distribution(pool_assembly, collapse=True)
            consistency, _ = pearsonr(single_categorical.values.flatten(), pool_categorical.values.flatten())
            uncorrected_consistencies.append(consistency)
            # Spearman-Brown correction for sub-sampling
            corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
            consistencies.append(corrected_consistency)
        consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
        uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
        average_consistency = consistencies.median('split')
        average_consistency.attrs['raw'] = consistencies
        average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
        return average_consistency


def Zhu2019ResponseMatch():
    return _Zhu2019ResponseMatch()


def Zhu2019Accuracy():
    return _Zhu2019Accuracy()
