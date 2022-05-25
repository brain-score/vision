import os

import numpy as np
import pytest
from brainio.assemblies import NeuroidAssembly, DataAssembly
from pytest import approx

from brainscore.metrics.rdm import RSA, RDMSimilarity, RDMMetric, RDMCrossValidated
from tests.test_metrics import load_hvm


class TestRDMCrossValidated:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'stimulus_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = RDMCrossValidated()
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score.sel(aggregation='center') == approx(1)


class TestRSA:
    @pytest.mark.private_access
    def test_equal_hvm(self):
        hvm = load_hvm().sel(region='IT')
        metric = RDMMetric()
        score = metric(hvm, hvm)
        assert score == approx(1.)

    @pytest.mark.skip(reason="loaded ordering likely incorrect")
    def test_hvm(self):
        hvm_it_v6_obj = self._load_neural_data()
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

    @pytest.mark.skip(reason="loaded ordering likely incorrect")
    def test_hvm_T(self):
        hvm_it_v6_obj = self._load_neural_data().T
        assert hvm_it_v6_obj.shape == (168, 64)
        self._test_hvm(hvm_it_v6_obj)

    def _test_hvm(self, hvm_it_v6_obj):
        loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")
        rsa_characterization = RSA()
        rsa = rsa_characterization(hvm_it_v6_obj)
        np.testing.assert_array_equal(rsa.shape, [64, 64])
        np.testing.assert_array_equal(rsa.dims, ['presentation', 'presentation'])
        assert rsa.values == approx(loaded, abs=1e-6)

    def _load_neural_data(self):
        data = load_hvm(group=lambda hvm: hvm.multi_groupby(["category_name", "object_name"]))
        data = data.sel(region='IT')
        return data

    def test_alignment(self):
        assembly = NeuroidAssembly([[1, 2], [1, 2], [4, 3], [4, 3]],
                                   coords={'stimulus_id': ('presentation', list(range(4))),
                                           'image_meta': ('presentation', list(range(4))),
                                           'neuroid_id': ('neuroid', list(range(2))),
                                           'neuroid_meta': ('neuroid', list(range(2)))},
                                   dims=['presentation', 'neuroid'])
        matrix = RSA()(assembly)
        assert np.all(np.diag(matrix) == approx(1., abs=.001))
        assert all(matrix.values[np.triu_indices(matrix.shape[0], k=1)] ==
                   matrix.values[np.tril_indices(matrix.shape[0], k=-1)]), "upper and lower triangular need to be equal"
        expected = DataAssembly([[1., 1., -1., -1.],
                                 [1., 1., -1., -1.],
                                 [-1., -1., 1., 1.],
                                 [-1., -1., 1., 1.]],
                                coords={'stimulus_id': ('presentation', list(range(4))),
                                        'image_meta': ('presentation', list(range(4)))},
                                dims=['presentation', 'presentation'])
        np.testing.assert_array_almost_equal(matrix.values, expected.values)  # does not take ordering into account


class TestRDMSimilarity(object):
    def test_2d_equal20(self):
        rdm = np.random.rand(20, 20)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'stimulus_id': ('presentation', list(range(20))),
                                           'object_name': ('presentation', ['A', 'B'] * 10)},
                              dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_2d_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'stimulus_id': ('presentation', list(range(100))),
                                           'object_name': ('presentation', ['A', 'B'] * 50)},
                              dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)


def test_np_load():
    p_path = os.path.join(os.path.dirname(__file__), "it_rdm.p")
    it_rdm = np.load(p_path, encoding="latin1", allow_pickle=True)
    assert it_rdm.shape == (64, 64)
