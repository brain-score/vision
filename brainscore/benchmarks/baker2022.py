import numpy as np
import numpy.random

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy_delta import AccuracyDelta, AccuracyDeltaCeiling
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{BAKER2022104913,
                title = {Deep learning models fail to capture the configural nature of human shape perception},
                journal = {iScience},
                volume = {25},
                number = {9},
                pages = {104913},
                year = {2022},
                issn = {2589-0042},
                doi = {https://doi.org/10.1016/j.isci.2022.104913},
                url = {https://www.sciencedirect.com/science/article/pii/S2589004222011853},
                author = {Nicholas Baker and James H. Elder},
                keywords = {Biological sciences, Neuroscience, Sensory neuroscience},
                abstract = {Summary
                A hallmark of human object perception is sensitivity to the holistic configuration of the local shape features of an object. Deep convolutional neural networks (DCNNs) are currently the dominant models for object recognition processing in the visual cortex, but do they capture this configural sensitivity? To answer this question, we employed a dataset of animal silhouettes and created a variant of this dataset that disrupts the configuration of each object while preserving local features. While human performance was impacted by this manipulation, DCNN performance was not, indicating insensitivity to object configuration. Modifications to training and architecture to make networks more brain-like did not lead to configural processing, and none of the networks were able to accurately predict trial-by-trial human object judgements. We speculate that to match human configural sensitivity, networks must be trained to solve a broader range of object tasks beyond category recognition.}
        }"""

DATASETS = ['normal', 'inverted']


class _Baker2022AccuracyDelta(BenchmarkBase):
    def __init__(self, dataset: str, image_types: list):
        self._metric = AccuracyDelta(image_types=image_types)
        self.image_types = image_types
        self.orientation = dataset
        self._ceiling = SplitHalvesConsistencyBaker(num_splits=100,
                                                    consistency_metric=AccuracyDeltaCeiling(self.image_types),
                                                    split_coordinate="subject", image_types=self.image_types)
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 8.8
        self._number_of_trials = 1

        super(_Baker2022AccuracyDelta, self).__init__(
            identifier=f'Baker2022{dataset}-accuracy_delta', version=1,
            ceiling_func=lambda: self._ceiling(assembly=self._assembly),
            parent='Baker2022',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        if self.orientation == "inverted":
            inverted_stimuli = stimulus_set[stimulus_set["orientation"] == "inverted"]
            labels = candidate.look_at(inverted_stimuli, number_of_trials=self._number_of_trials)
            inverted_assembly = self._assembly[self._assembly["orientation"] == "inverted"]
            raw_score = self._metric(labels, inverted_assembly)
            ceiling = self._ceiling(inverted_assembly)
        else:
            labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
            raw_score = self._metric(labels, self._assembly)
            ceiling = self._ceiling(self._assembly)
        score = raw_score / ceiling

        # cap score at 1 if ceiled score > 1
        score[0] = 1 if score[0] > 1 else score[0]

        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def Baker2022AccuracyDeltaFrankenstein():
    return _Baker2022AccuracyDelta(dataset='normal', image_types=["w", "f"])


def Baker2022AccuracyDeltaFragmented():
    return _Baker2022AccuracyDelta(dataset='normal', image_types=["w", "o"])


def Baker2022InvertedAccuracyDelta():
    return _Baker2022AccuracyDelta(dataset='inverted', image_types=["w", "f"])


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'Baker2022_{dataset}_distortion')
    return assembly


# ceiling method:
class SplitHalvesConsistencyBaker:
    def __init__(self, num_splits: int, split_coordinate: str, consistency_metric, image_types):
        """
        :param num_splits: how many times to create two halves
        :param split_coordinate: over which coordinate to split the assembly into halves
        :param consistency_metric: which metric to use to compute the consistency of two halves
        """
        self.num_splits = num_splits
        self.split_coordinate = split_coordinate
        self.consistency_metric = consistency_metric
        self.image_types = image_types

    def __call__(self, assembly) -> Score:

        consistencies, uncorrected_consistencies = [], []
        splits = range(self.num_splits)
        random_state = np.random.RandomState(0)
        for _ in splits:
            num_subjects = len(set(assembly["subject"].values))
            half1_subjects = random_state.choice(range(1, num_subjects), (num_subjects // 2), replace=False)
            half1 = assembly[
                {'presentation': [subject in half1_subjects for subject in assembly['subject'].values]}]
            half2 = assembly[
                {'presentation': [subject not in half1_subjects for subject in assembly['subject'].values]}]
            consistency = self.consistency_metric(half1, half2)
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
