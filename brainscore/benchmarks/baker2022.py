import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy_delta import AccuracyDelta
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from brainscore.metrics.ceiling import SplitHalvesConsistencyBaker

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

# create functions so that users can import individual benchmarks as e.g. Baker2022wholeAccuracyDelta
for dataset in DATASETS:
    # normal experiment
    identifier = f"Baker2022{dataset.replace('_', '')}AccuracyDelta"
    globals()[identifier] = lambda dataset=dataset: _Baker2022AccuracyDelta(dataset)

    # # inverted experiment
    # identifier = f"Baker2022_Inverted{dataset.replace('_', '')}AccuracyDelta"
    # globals()[identifier] = lambda dataset=dataset: _Baker2022InvertedAccuracyDelta(dataset)


class _Baker2022AccuracyDelta(BenchmarkBase):

    def __init__(self, dataset: str, image_types: list):
        self._metric = AccuracyDelta()

        # image types: list[str]. Either ["w", "f"] for frankenstein delta or ["w", "o"] for fragmented delta.
        self.image_types = image_types
        self.orientation = dataset

        self._ceiling = SplitHalvesConsistencyBaker(num_splits=100, consistency_metric=AccuracyDelta(),
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
            raw_score = self._metric(labels, inverted_assembly, image_types=self.image_types)
            ceiling, ceiling_error = self._ceiling(inverted_assembly)
        else:
            labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
            raw_score = self._metric(labels, self._assembly, image_types=self.image_types)
            ceiling, ceiling_error = self._ceiling(self._assembly)
        score = raw_score / ceiling
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
    assembly = brainscore.get_assembly(f'kellmen.Baker2022_{dataset}_distortion')
    return assembly