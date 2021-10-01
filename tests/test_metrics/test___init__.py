import numpy as np
from brainio.assemblies import DataAssembly

from brainscore.metrics import Score


class TestScoreRaw:
    def test_sel(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        sel_score = score.sel(a=1)
        np.testing.assert_array_equal(sel_score.raw['a'], [1, 1])

    def test_isel(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        sel_score = score.isel(a=0)
        np.testing.assert_array_equal(sel_score.raw['a'], [1, 1])

    def test_sel_no_apply_raw(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        sel_score = score.sel(a=1, _apply_raw=False)
        np.testing.assert_array_equal(sel_score.raw['a'], [1, 1, 2, 2])

    def test_squeeze(self):
        score = Score([[1, 2]], coords={'s': [0], 'a': [1, 2]}, dims=['s', 'a'])
        score.attrs['raw'] = DataAssembly([[0, 2, 1, 3]], coords={'s': [0], 'a': [1, 1, 2, 2]}, dims=['s', 'a'])
        sel_score = score.squeeze('s')
        np.testing.assert_array_equal(sel_score.raw.dims, ['a'])

    def test_mean(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        mean_score = score.mean('a')
        np.testing.assert_array_equal(mean_score.raw['a'], [1, 1, 2, 2])

    def test_mean_no_apply_raw(self):
        score = Score([1, 2], coords={'a': [1, 2]}, dims=['a'])
        score.attrs['raw'] = DataAssembly([0, 2, 1, 3], coords={'a': [1, 1, 2, 2]}, dims=['a'])
        mean_score = score.mean('a', _apply_raw=True)
        assert mean_score.raw == 1.5
