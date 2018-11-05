import numpy as np
import pytest

from brainscore.assemblies import DataAssembly


class TestConstructor:
    def test_multi_coord_dim(self):
        d = DataAssembly([0, 1], coords={'coord1': ('dim', [0, 1]), 'coord2': ('dim', [1, 1])}, dims=['dim'])
        np.testing.assert_array_equal(d.sel(coord1=0).values, [0])
        np.testing.assert_array_equal(d.sel(coord2=1).values, [0, 1])

    def test_single_coord_dim(self):
        d = DataAssembly([0, 1], coords={'coord1': ('dim', [0, 1])}, dims=['dim'])
        np.testing.assert_array_equal(d.sel(coord1=0).values, [0])

    def test_multi_dims(self):
        d = DataAssembly([[0, 1], [2, 3]],
                         coords={
                             'coord1_1': ('dim1', [0, 1]), 'coord1_2': ('dim1', [1, 1]),
                             'coord2_1': ('dim2', [0, 1]), 'coord2_2': ('dim2', [1, 1])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1_1=0, coord2_1=0).values, [[0]])
        np.testing.assert_array_equal(d.sel(coord1_2=1, coord2_2=1).values, [[0, 1], [2, 3]])
        np.testing.assert_array_equal(d.sel(coord1_1=0).values, [[0, 1]])
        np.testing.assert_array_equal(d.sel(coord2_2=1).values, [[0, 1], [2, 3]])

    def test_multi_dims_single_coord_unique(self):
        d = DataAssembly([[0, 1]],
                         coords={'coord1': ('dim1', [0]), 'coord2': ('dim2', [1, 2])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1=0, coord2=1).values, [[0]])
        np.testing.assert_array_equal(d.sel(coord1=0, coord2=2).values, [1])  # ideally this would be [[1]]

    def test_multi_dims_single_coord_unique_nonunique(self):
        d = DataAssembly([[0, 1]],
                         coords={'coord1': ('dim1', [0]), 'coord2': ('dim2', [1, 1])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1=0, coord2=1).values, [0, 1])  # ideally this would be [[0, 1]]

    def test_multi_dims_single_coord_nonunique(self):
        d = DataAssembly([[0, 1], [2, 3]],
                         coords={'coord1': ('dim1', [1, 1]), 'coord2': ('dim2', [1, 1])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1=1, coord2=1).values, [[0, 1], [2, 3]])

    def test_multi_dims_mixed_coord_unique(self):
        d = DataAssembly([[0, 1], [2, 3]],
                         coords={'coord1_1': ('dim1', [0, 1]), 'coord1_2': ('dim1', [1, 1]),
                                 'coord2': ('dim2', [0, 1])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1_1=0, coord2=0).values, [0])  # ideally this would be [[0]]
        np.testing.assert_array_equal(d.sel(coord1_2=1, coord2=1).values, [1, 3])  # ideally this would be [[1], [3]]

    def test_multi_dims_mixed_coord_nonunique(self):
        d = DataAssembly([[0, 1], [2, 3]],
                         coords={'coord1_1': ('dim1', [0, 1]), 'coord1_2': ('dim1', [1, 1]),
                                 'coord2': ('dim2', [1, 1])},
                         dims=['dim1', 'dim2'])
        np.testing.assert_array_equal(d.sel(coord1_2=1, coord2=1).values, [[0, 1], [2, 3]])


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
