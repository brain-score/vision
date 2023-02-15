import numpy as np

from brainscore.metrics import Score


class ResponseMatch:
    def __call__(self, source, target):
        match = source.values == target.values
        center = np.mean(match)
        error = np.std(match)
        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = match
        return score
