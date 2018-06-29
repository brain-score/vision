import os

import numpy as np
import pytest
from pytest import approx

from brainscore.assemblies import NeuroidAssembly
from brainscore.metrics.rdm import RSA, RDMSimilarity, RDMMetric
from tests.test_metrics import load_hvm


class TestRSA(object):
    def test_equal_hvm(self):
        hvm = load_hvm().sel(region='IT')
        metric = RDMMetric()
        score = metric(hvm, hvm)
        assert score == approx(1.)

    def test_hvm(self):
        hvm_it_v6_obj = self._load_neural_data()
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

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


class TestRDMSimilarity(object):
    def test_2d_equal20(self):
        rdm = np.random.rand(20, 20)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(
            rdm, coords={'presentation': list(range(20)), 'object_name': ('presentation', ['A', 'B'] * 10)},
            dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)

    def test_2d_equal100(self):
        rdm = np.random.rand(100, 100)  # not mirrored across diagonal, but fine for unit test
        np.fill_diagonal(rdm, 0)
        rdm = NeuroidAssembly(
            rdm, coords={'presentation': list(range(100)), 'object_name': ('presentation', ['A', 'B'] * 50)},
            dims=['presentation', 'presentation'])
        similarity = RDMSimilarity()
        score = similarity(rdm, rdm)
        assert score == approx(1.)


def test_np_load():
    p_path = os.path.join(os.path.dirname(__file__), "it_rdm.p")
    it_rdm = np.load(p_path, encoding="latin1")
    assert it_rdm.shape == (64, 64)
