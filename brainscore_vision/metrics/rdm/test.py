import numpy as np
from pytest import approx

from brainio.assemblies import NeuroidAssembly, DataAssembly
from brainscore_vision import load_metric
from .metric import RDM


class TestCharacterization:
    def test_alignment(self):
        assembly = NeuroidAssembly([[1, 2], [1, 2], [4, 3], [4, 3]],
                                   coords={'stimulus_id': ('presentation', list(range(4))),
                                           'image_meta': ('presentation', list(range(4))),
                                           'neuroid_id': ('neuroid', list(range(2))),
                                           'neuroid_meta': ('neuroid', list(range(2)))},
                                   dims=['presentation', 'neuroid'])
        matrix = RDM()(assembly)
        assert np.all(np.diag(matrix) == approx(0, abs=.001))
        assert all(matrix.values[np.triu_indices(matrix.shape[0], k=1)] ==
                   matrix.values[np.tril_indices(matrix.shape[0], k=-1)]), "upper and lower triangular need to be equal"
        expected = DataAssembly([[0, 0, 2, 2],
                                 [0, 0, 2, 2],
                                 [2, 2, 0, 0],
                                 [2, 2, 0, 0]],
                                coords={'stimulus_id': ('presentation', list(range(4))),
                                        'image_meta': ('presentation', list(range(4)))},
                                dims=['presentation', 'presentation'])
        np.testing.assert_array_almost_equal(matrix.values, expected.values)  # does not take ordering into account


class TestRDMCrossValidated:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = load_metric('rdm_cv')
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score == approx(1)


class TestRDMSimilarity:
    def test_2d_equal20(self):
        rdm = np.random.rand(20, 20)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'stimulus_id': ('presentation', list(range(20))),
                                           'object_name': ('presentation', ['A', 'B'] * 10)},
                              dims=['presentation', 'presentation'])
        similarity = load_metric('rdm')
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_2d_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'stimulus_id': ('presentation', list(range(100))),
                                           'object_name': ('presentation', ['A', 'B'] * 50)},
                              dims=['presentation', 'presentation'])
        similarity = load_metric('rdm')
        score = similarity(rdm, rdm)
        assert score == approx(1.)
