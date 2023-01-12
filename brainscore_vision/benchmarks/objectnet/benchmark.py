import os
import logging

import numpy as np
import pandas as pd

import brainscore_vision
from brainio.stimuli import StimulusSet
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.metrics.accuracy import Accuracy
from brainscore_vision.model_interface import BrainModel
from brainio.fetch import StimulusSetLoader
from brainio.lookup import lookup_stimulus_set

NUMBER_OF_TRIALS = 10

_logger = logging.getLogger(__name__)
LOCAL_STIMULUS_DIRECTORY = '/braintree/data2/active/common/objectnet-stimuli/'


class Objectnet(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = brainscore_vision.load_stimulus_set('katz.BarbuMayo2019')
        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
       
        super(Objectnet, self).__init__(identifier='katz.BarbuMayo2019-top1', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='engineering',
                                           bibtex="""@inproceedings{DBLP:conf/nips/BarbuMALWGTK19,
                                                    author    = {Andrei Barbu and
                                                                David Mayo and
                                                                Julian Alverio and
                                                                William Luo and
                                                                Christopher Wang and
                                                                Dan Gutfreund and
                                                                Josh Tenenbaum and
                                                                Boris Katz},
                                                    title     = {ObjectNet: {A} large-scale bias-controlled dataset for pushing the
                                                                limits of object recognition models},
                                                    booktitle = {NeurIPS 2019},
                                                    pages     = {9448--9458},
                                                    year      = {2019},
                                                    url       = {https://proceedings.neurips.cc/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html},
                                                    }""")

    def __call__(self, candidate):
        # The proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # For now, since almost all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, 'imagenet')
        stimulus_set = self._stimulus_set[list(set(self._stimulus_set.columns) - {'synset'})]  # do not show label
        predictions = candidate.look_at(stimulus_set, number_of_trials=NUMBER_OF_TRIALS)
        score = self._similarity_metric(
            predictions.sortby('filename'),
            self._stimulus_set.sort_values('filename')['synset'].values
        )
        return score
