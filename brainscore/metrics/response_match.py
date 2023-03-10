import numpy as np

from brainio.assemblies import DataAssembly
from brainscore.metrics import Score, Metric


class ResponseMatch(Metric):
    def __call__(self, source: DataAssembly, target: DataAssembly) -> Score:
        match = source.values == target.values
        center = np.mean(match)
        error = np.std(match)
        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = match
        return score
