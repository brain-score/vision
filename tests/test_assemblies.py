from brainscore.assemblies import DataAssembly


class TestMultiGroupby:
    def test_single_dimension(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': ['a', 'b'], 'b': ['x', 'y', 'z']}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean()
        assert g.equals(DataAssembly([2, 5], coords={'a': ['a', 'b']}, dims=['a']))

    def test_single_dimension_int(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6]], coords={'a': [1, 2], 'b': [3, 4, 5]}, dims=['a', 'b'])
        g = d.multi_groupby(['a']).mean()
        assert g.equals(DataAssembly([2, 5], coords={'a': [1, 2]}, dims=['a']))

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


class TestMultiDimGroupby:
    def test_unique_values(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['a', 'b'], lambda x, **_: x)
        assert g.equals(d)

    def test_unique_values_swappeddims(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['b', 'a'], lambda x, **_: x)
        assert g.equals(d)

    def test_subtract_mean(self):
        d = DataAssembly([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                         coords={'a': ['a', 'b', 'c', 'd'],
                                 'aa': ('a', ['a', 'a', 'b', 'b']),
                                 'b': ['x', 'y', 'z']},
                         dims=['a', 'b'])
        g = d.multi_dim_apply(['aa', 'b'], lambda x, **_: x - x.mean())
        assert g.equals(DataAssembly([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]],
                                     coords={'a': ['a', 'b', 'c', 'd'],
                                             'aa': ('a', ['a', 'a', 'b', 'b']),
                                             'b': ['x', 'y', 'z']},
                                     dims=['a', 'b']))
