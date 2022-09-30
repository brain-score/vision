import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{zhu2019robustness,
            title={Robustness of object recognition under extreme occlusion in humans and computational models},
            author={Zhu, Hongru and Tang, Peng and Park, Jeongho and Park, Soojin and Yuille, Alan},
            journal={arXiv preprint arXiv:1905.04598},
            year={2019}
        }"""

DATASETS = ['extreme_occlusion']

# create functions so that users can import individual benchmarks as e.g. Geirhos2021sketchErrorConsistency
for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Zhu2021{dataset.replace('-', '')}RDM"
    globals()[identifier] = lambda dataset=dataset: _Zhu2019RDM(dataset)


class _Zhu2019RDM(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = RDM()
        self._assembly = LazyLoad(lambda: load_assembly(dataset))
        self._visual_degrees = 8

        self._number_of_trials = 1

        super(_Zhu2019RDM, self).__init__(
            identifier=f'yuille.Zhu2019_{dataset}-RDM', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='yuille.Zhu2019',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def load_assembly(dataset):
    assembly = brainscore.get_assembly(f'yuille.Zhu2019_{dataset}')
    return assembly
