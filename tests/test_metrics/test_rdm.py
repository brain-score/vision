import os

import numpy as np
from pytest import approx

from mkgu.metrics.rdm import RSA, RDMMetric
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


class TestRDMMetric(object):
    def test_equal(self):
        hvm = load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
        rdm_metric = RDMMetric()
        score = rdm_metric(hvm, hvm)
        assert score == 1.
