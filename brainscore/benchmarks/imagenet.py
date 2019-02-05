import pandas as pd

from brainscore.benchmarks import Benchmark
from brainscore.metrics.accuracy import Accuracy
from brainio_base.stimuli import StimulusSet


class Imagenet2012(Benchmark):
    def __init__(self):
        stimulus_set = pd.read_csv('/braintree/home/msch/brainio_contrib/mkgu_packaging/imagenet/'
                                   'imagenet2012.csv')
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.image_id: row.filepath for row in stimulus_set.itertuples()}
        self._stimulus_set = stimulus_set

        self._name = 'imagenet2012'
        self._similarity_metric = Accuracy()

    @property
    def name(self):
        return self._name

    def __call__(self, candidate):
        predictions = candidate(self._stimulus_set)
        score = self._similarity_metric(predictions, self._stimulus_set['label'].values)
        return score

    @property
    def ceiling(self):
        return None
