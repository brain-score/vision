import os

import pandas as pd

from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks import Benchmark
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel


class Imagenet2012(Benchmark):
    def __init__(self):
        stimulus_set = pd.read_csv(os.path.join(os.path.dirname(__file__), 'imagenet2012.csv'))
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.image_id: row.filepath for row in stimulus_set.itertuples()}
        self._stimulus_set = stimulus_set

        self._name = 'imagenet2012'
        self._similarity_metric = Accuracy()

    @property
    def name(self):
        return self._name

    def __call__(self, candidate):
        # the proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # for now, since all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, 'imagenet')
        predictions = candidate.look_at(self._stimulus_set[list(set(self._stimulus_set.columns) - {'synset'})])
        score = self._similarity_metric(predictions, self._stimulus_set['synset'].values)
        return score

    @property
    def ceiling(self):
        return None
