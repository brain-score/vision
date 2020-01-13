import os

import numpy as np
import pandas as pd

from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel


class Imagenet2012(BenchmarkBase):
    def __init__(self):
        stimulus_set = pd.read_csv(os.path.join(os.path.dirname(__file__), 'imagenet2012.csv'))
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.image_id: row.filepath for row in stimulus_set.itertuples()}
        self._stimulus_set = stimulus_set
        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet2012, self).__init__(identifier='fei-fei.Deng2009-top1', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='ImageNet',
                                           paper_link="https://ieeexplore.ieee.org/abstract/document/5206848")

    def __call__(self, candidate):
        # the proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # for now, since all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, 'imagenet')
        predictions = candidate.look_at(self._stimulus_set[list(set(self._stimulus_set.columns) - {'synset'})])
        score = self._similarity_metric(predictions, self._stimulus_set['synset'].values)
        return score
