import pytest

from brainscore.assemblies import DataAssembly


class TestMultiGroupby:
    def test_single_dimension(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean()
        assert g.equals(DataAssembly([2, 5], coords={'a': ['a', 'b']}, dims=['a']))

    def test_single_coord(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]],
                         coords={'a': ('multi_dim', ['a', 'b']), 'b': ('multi_dim', ['c', 'c']), 'c': ['x', 'y', 'z']},
                         dims=['multi_dim', 'c'])
        g = d.multi_groupby(['a']).mean()
        assert g.equals(DataAssembly([2, 5], coords={'multi_dim': ['a', 'b']}, dims=['multi_dim']))
        # ideally, we would want `g.equals(DataAssembly([2, 5],
        #   coords={'a': ('multi_dim', ['a', 'b']), 'b': ('multi_dim', ['c', 'c'])}, dims=['multi_dim']))`
        # but this is fine for now.

    def test_single_dim_multi_coord(self):
        d = DataAssembly([1, 2, 3, 4, 5, 6],
                         coords={'a': ('multi_dim', ['a', 'a', 'a', 'a', 'a', 'a']),
                                 'b': ('multi_dim', ['a', 'a', 'a', 'b', 'b', 'b']),
                                 'c': ('multi_dim', ['a', 'b', 'c', 'd', 'e', 'f'])},
                         dims=['multi_dim'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2, 5],
                                     coords={'a': ('multi_dim', ['a', 'a']), 'b': ('multi_dim', ['a', 'b'])},
                                     dims=['multi_dim']))

    @pytest.mark.skip(reason="not implemented")
    def test_multi_dim(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'a', 'b', 'b'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_groupby(['a', 'b']).mean()
        assert g.equals(DataAssembly([2.5, 3.5, 4.5], [8.5, 9.5, 10.5],
                                     coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']},
                                     dims=['a', 'b']))


class TestMultiDimGroupby:
    def test_unique_values(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_groupby(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

    def test_nonunique_singledim(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'a', 'b', 'b'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_groupby(['a', 'b'], lambda x, **_: x.mean())
        assert g.equals(DataAssembly([2.5, 3.5, 4.5], [8.5, 9.5, 10.5],
                                     coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']},
                                     dims=['a', 'b']))
