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


class _Zhu2019Accuracy(BenchmarkBase):
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
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                           source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
        label_predictions = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        labels = get_choices(label_predictions, categories=["car", "aeroplane", "motorbike", "bicycle", "bus"])
        score = self._metric(labels, target=self._stimulus_set['ground_truth'].values)
        return score


# takes 5-way probability vector and returns category
def get_choices(predictions, categories):
    choices = []
    for prediction in predictions:
        choice_index = list(prediction.values).index(max(prediction))
        choice = categories[choice_index]
        choices.append(choice)
    return np.array(choices)


def Zhu2019Accuracy():
    return _Zhu2019Accuracy(dataset='extreme_occlusion')


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'yuille.Zhu2019_{dataset}')
    return assembly
