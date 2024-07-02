import numpy as np
from brainscore_core import Metric
from brainscore_vision.metrics import Score

'''
This metric takes in two floats and gives a score between 0 and 1 based on 
how far apart the numbers are, using a sigmoid function.

'''


class ValueDelta(Metric):

    def __init__(self, scale: float = 1.00):
        """

        :param scale: float, the "steepness" of the sigmoid curve. With a high scale value (> ~1.5), even small
        differences will result in a significantly lower score. With a low scale value (< ~0.5), the score will remain
        relatively high for larger differences. 1.0 was chosen as default due to being a reasonable middle ground.
        """
        self.scale = scale

    def __call__(self, source_value: float, target_value: float) -> Score:
        abs_diff = float(np.abs(source_value - target_value))
        center = 1 / (np.exp(self.scale * abs_diff))
        center = min(max(center, 0), 1)
        score = Score(center)
        score.attrs['error'] = np.nan
        score.attrs[Score.RAW_VALUES_KEY] = [source_value, target_value]
        return score
