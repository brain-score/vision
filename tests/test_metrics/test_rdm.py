import os

import numpy as np
from pytest import approx

from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics.rdm import RSA, RDMMetric, RDMCorrelationCoefficient
from tests.test_metrics import load_hvm


class TestRDM(object):
    def test_hvm(self):
        hvm_it_v6_obj = load_hvm(group=lambda hvm: hvm.multi_groupby(["category_name", "object_name"]))
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

    def test_hvm_T(self):
        hvm_it_v6_obj = load_hvm(group=lambda hvm: hvm.multi_groupby(["category_name", "object_name"])).T
        assert hvm_it_v6_obj.shape == (168, 64)
        self._test_hvm(hvm_it_v6_obj)

    def _test_hvm(self, hvm_it_v6_obj):
        loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")
        rsa_characterization = RSA()
        rsa = rsa_characterization(hvm_it_v6_obj)
        assert list(rsa.shape) == [64, 64]
        assert list(rsa.dims) == ['presentation', 'presentation']
        assert rsa.values == approx(loaded, abs=1e-6)


class TestRDMSimilarity(object):
    def test_2d_equal5(self):
        rdm = np.random.rand(5, 5)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'presentation': list(range(5))}, dims=['presentation', 'presentation'])
        similarity = RDMCorrelationCoefficient()
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_2d_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'presentation': list(range(100))}, dims=['presentation', 'presentation'])
        similarity = RDMCorrelationCoefficient()
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_3d_equal(self):
        values = np.broadcast_to(np.random.rand(5, 5, 1), [5, 5, 3]).copy()
        diag_indices = np.diag_indices(5)
        values[diag_indices] = 0
        assembly1 = NeuroidAssembly(values, coords={'presentation': list(range(5)), 'dim1': list(range(3))},
                                    dims=['presentation', 'presentation', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={'presentation': list(range(5)), 'dim2': list(range(3))},
                                    dims=['presentation', 'presentation', 'dim2'])
        similarity = RDMCorrelationCoefficient()
        scores = similarity(assembly1, assembly2)
        np.testing.assert_array_equal(scores.shape, [3, 3])
        np.testing.assert_array_almost_equal(scores, np.broadcast_to(1, [3, 3]))

    def test_3d_diag(self):
        values = np.random.rand(5, 5, 3)
        diag_indices = np.diag_indices(5)
        values[diag_indices] = 0
        assembly1 = NeuroidAssembly(values, coords={'presentation': list(range(5)), 'dim1': list(range(3))},
                                    dims=['presentation', 'presentation', 'dim1'])
        assembly2 = NeuroidAssembly(values, coords={'presentation': list(range(5)), 'dim2': list(range(3))},
                                    dims=['presentation', 'presentation', 'dim2'])
        similarity = RDMCorrelationCoefficient()
        scores = similarity(assembly1, assembly2)
        np.testing.assert_array_equal(scores.shape, [3, 3])
        np.testing.assert_array_almost_equal(np.diag(scores), np.broadcast_to(1, [3]))

    def test_3d_equal_presentation_last(self):
        values = np.broadcast_to(np.random.rand(5, 5, 1), [5, 5, 3]).copy()
        values[np.diag_indices(5)] = 0
        assembly1 = NeuroidAssembly(values.T, coords={'presentation': list(range(5)), 'dim1': list(range(3))},
                                    dims=['dim1', 'presentation', 'presentation'])
        assembly2 = NeuroidAssembly(values.T, coords={'presentation': list(range(5)), 'dim2': list(range(3))},
                                    dims=['dim2', 'presentation', 'presentation'])
        similarity = RDMCorrelationCoefficient()
        scores = similarity(assembly1, assembly2)
        np.testing.assert_array_equal(scores.shape, [3, 3])
        np.testing.assert_array_almost_equal(scores, np.broadcast_to(1, [3, 3]))


class TestRDMMetric(object):
    def test_equal(self):
        hvm = load_hvm(group=lambda hvm: hvm.multi_groupby(["category_name", "object_name"]))
        rdm_metric = RDMMetric()
        score = rdm_metric(hvm, hvm)
        assert score == approx(1.)


def test_np_load():
    print(os.getcwd())
    p_path = os.path.join(os.path.dirname(__file__), "it_rdm.p")
    it_rdm = np.load(p_path, encoding="latin1")
    print(it_rdm)
    assert it_rdm.shape == (64, 64)
