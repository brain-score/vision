import numpy as np
import os
import json

from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel


class PhysionGlobalPredictionAccuracy(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
                                                        'physion_readout_all.json')),
                                      'r')

        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionAccuracy, self).__init__(identifier='Physionv1.5-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        candidate.start_task(BrainModel.Task.label, self._stimulus_set['train'])
        predictions = candidate.look_at(self._stimulus_set['test'])
        score = self._similarity_metric(
            predictions,
            stimulus_set['test']['labels']
        )
        return score

class PhysionSnippetPredictionAccuracy(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
                                                        'physion_readout_all.json')),
                                      'r')

        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionAccuracy, self).__init__(identifier='Physionv1.5-snippet-simulation-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        candidate.start_task(BrainModel.Task.label, self._stimulus_set['train'])
        predictions = candidate.look_at(self._stimulus_set['test'])
        score = self._similarity_metric(
            predictions,
            stimulus_set['test']['labels']
        )
        return score

class PhysionGlobalGeneralization(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
                                                        'physion_readout_intra_scenario.json')),
                                      'r')

        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionAccuracy, self).__init__(identifier='Physionv1.5-intra-scenario-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        candidate.start_task(BrainModel.Task.label, self._stimulus_set['train'])
        predictions = candidate.look_at(self._stimulus_set['test'])
        score = self._similarity_metric(
            predictions,
            stimulus_set['test']['labels']
        )
        return score

class PhysionSnippetDetectionAccuracy(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
                                                        'physion_readout_all.json')),
                                      'r')

        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionAccuracy, self).__init__(identifier='Physionv1.5-snippet-rollout-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        candidate.start_task(BrainModel.Task.label, self._stimulus_set['train'])
        predictions = candidate.look_at(self._stimulus_set['test'])
        score = self._similarity_metric(
            predictions,
            stimulus_set['test']['labels']
        )
        return score
