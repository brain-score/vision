import numpy as np

from brainscore_core import Metric
from brainscore_vision.metrics import Score


class Accuracy(Metric):
    def __call__(self, source, target) -> Score:
        values = source == target
        center = np.mean(values)
        error = np.std(values)

        score = Score(center)
        score.attrs['error'] = error
        score.attrs[Score.RAW_VALUES_KEY] = values
        return score
