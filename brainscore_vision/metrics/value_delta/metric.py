import numpy as np

from brainscore_core import Metric
from brainscore_vision.metrics import Score


class ValueDelta(Metric):
    def __call__(self, source_value: float, target_value: float) -> Score:
        center = max((1 - (np.abs(source_value - target_value) / (1 - 0))), 0)
        score = Score(center)
        score.attrs['error'] = 0.00
        score.attrs[Score.RAW_VALUES_KEY] = [source_value, target_value]
        return score
