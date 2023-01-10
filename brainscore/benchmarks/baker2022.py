import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy_delta import AccuracyDelta
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from brainscore.metrics.ceiling import SplitHalvesConsistencyBaker



'''
Made to directly address statement by Bowers et al. 2022:

"Nevertheless, when DNNs classify objects based on shape, they use the wrong sort of shape representations.  
For instance, in contrast with a large body of research showing that humans tend to rely on the global shape of 
objects, Baker, Lu, Erlikhman, and Kellman (2018) showed that DNNs focus on local shape features.  That is, they 
found that DNNs trained on ImageNet could correctly classify some silhouette images (where all diagnostic texture 
information was removed), indicating that these images were identified based on shape.  However, when the 
local shape features of the silhouettes were disrupted by including jittered contours, the models did much more poorly.
By contrast, DNNs were more successful when the parts of the silhouettes were rearranged, a manipulation that 
kept many local shape features but disrupted the overall shape.  Humans show the opposite pattern. "

In other words:

Local Disruption = Humans Ok, Models Bad
Global Disruption = Humans Bad, Models Ok

Baker 2022 tests for pictures that disrupt global features, but keep local features. That is, models should 
do much more poorly on these images then humans. 

'''

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

# create functions so that users can import individual benchmarks as e.g. Baker2022wholeAboveChanceAgreement
for dataset in DATASETS:
    # normal experiment
    identifier = f"Baker2022{dataset.replace('_', '')}AccuracyDelta"
    globals()[identifier] = lambda dataset=dataset: _Baker2022AccuracyDelta(dataset)

    # # inverted experiment
    # identifier = f"Baker2022_Inverted{dataset.replace('_', '')}AccuracyDelta"
    # globals()[identifier] = lambda dataset=dataset: _Baker2022InvertedAccuracyDelta(dataset)


class _Baker2022AccuracyDelta(BenchmarkBase):

    def __init__(self, dataset, image_types):
        self._metric = AccuracyDelta()
        self.image_types = image_types
        self._ceiling = SplitHalvesConsistencyBaker(num_splits=100, consistency_metric=AccuracyDelta(),
                                               split_coordinate="subject", image_types=self.image_types)
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 8.8
        self._number_of_trials = 1

        super(_Baker2022AccuracyDelta, self).__init__(
            identifier=f'kellmen.Baker2022{dataset}-accuracy_delta', version=1,
            ceiling_func=lambda: self._ceiling(assembly=self._assembly),
            parent='kellmen.Baker2022',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly, image_types=self.image_types)
        ceiling = self._ceiling(self._assembly)
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def Baker2022AccuracyDeltaFrankenstein():
    return _Baker2022AccuracyDelta(dataset='normal', image_types=["w", "f"])


def Baker2022AccuracyDeltaFragmented():
    return _Baker2022AccuracyDelta(dataset='normal', image_types=["w", "o"])


"""
Inverted Benchmark.
This has 12 subjects, who saw 4 types of images in the combinations of the sets:
{normal, inverted} and {whole, frankenstein}

"""


class _Baker2022InvertedAboveChanceAgreement(BenchmarkBase):

    def __init__(self, dataset):
        self._metric = AboveChanceAgreement()
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 8

        self._number_of_trials = 1

        super(_Baker2022InvertedAboveChanceAgreement, self).__init__(
            identifier=f'kellmen.Baker2022_Inverted{dataset}-above_chance_agreement', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='kellmen.Baker2022',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self._ceiling(self._assembly)
        score = raw_score / ceiling
        # score.attrs['raw'] = raw_score
        # score.attrs['ceiling'] = ceiling
        return score


def Baker2022InvertedAboveChanceAgreement():
    return _Baker2022InvertedAboveChanceAgreement(dataset='inverted')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'kellmen.Baker2022_{dataset}_distortion')
    return assembly
