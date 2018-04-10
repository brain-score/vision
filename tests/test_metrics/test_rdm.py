import os

import numpy as np
from pytest import approx

from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics.rdm import RSA, RDMMetric, RDMCorrelationCoefficient
from tests.test_metrics import load_hvm


class TestRDM(object):
    def test_hvm(self):
        hvm_it_v6_obj = load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

    def test_hvm_T(self):
        hvm_it_v6_obj = load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"])).T
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
    def test_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'presentation': list(range(100))}, dims=['presentation', 'presentation'])
        similarity = RDMCorrelationCoefficient()
        score = similarity(rdm, rdm)
        assert score == 1.

    def test_equal5(self):
        rdm = np.random.rand(5, 5)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(rdm, coords={'presentation': list(range(5))}, dims=['presentation', 'presentation'])
        similarity = RDMCorrelationCoefficient()
        score = similarity(rdm, rdm)
        assert score == 1.

    def test_equal_3d(self):
        values = np.random.rand(5, 5, 3)
        diag_indices = np.diag_indices(5)
        values[diag_indices] = 0
        assembly = NeuroidAssembly(values, coords={'presentation': list(range(5)), '3rd_dim': list(range(3))},
                                   dims=['presentation', 'presentation', '3rd_dim'])
        similarity = RDMCorrelationCoefficient()
        scores = similarity(assembly, assembly)
        np.testing.assert_almost_equal(scores.values, [1., 1., 1.])

    def test_equal_4d(self):
        values = np.random.rand(5, 5, 3, 2)
        diag_indices = np.diag_indices(5)
        values[diag_indices] = 0
        assembly = NeuroidAssembly(values, coords={'presentation': list(range(5)),
                                                   '3rd_dim': list(range(3)), '4th_dim': list(range(2))},
                                   dims=['presentation', 'presentation', '3rd_dim', '4th_dim'])
        similarity = RDMCorrelationCoefficient()
        scores = similarity(assembly, assembly)
        np.testing.assert_array_equal(scores.shape, [3, 2])
        np.testing.assert_almost_equal(scores.values, [[1., 1.], [1., 1.], [1., 1.]])


class TestRDMMetric(object):
    def test_equal(self):
        hvm = load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
        rdm_metric = RDMMetric()
        score = rdm_metric(hvm, hvm)
        assert score == 1.
