import numpy as np

from brainscore.assemblies import DataAssembly
from brainscore.metrics import Score


class TestScoreRaw:
    def test_sel(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        sel_score = score.sel(a=1)
        np.testing.assert_array_equal(sel_score.attrs['raw']['a'], [1, 1])

    def test_mean_preserve(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        mean_score = score.mean('a')
        np.testing.assert_array_equal(mean_score.attrs['raw']['a'], [1, 1, 2, 2])
