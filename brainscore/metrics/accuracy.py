import numpy as np

from brainscore.metrics import Score


class Accuracy:
    def __call__(self, source, target):
        values = source == target
        center = np.mean(values)
        error = np.std(values)

        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = values
        return score
