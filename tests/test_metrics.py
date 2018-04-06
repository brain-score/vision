#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mkgu
----------------------------------

Tests for `mkgu` module.
"""

import os

import numpy as np
import xarray
from pytest import approx

import mkgu
from mkgu.metrics.neural_fit import NeuralFitMetric, PCANeuroidCharacterization
from mkgu.metrics.rdm import RSA, RDMMetric


class TestRDM(object):
    def test_hvm(self):
        hvm_it_v6_obj = _load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
        assert hvm_it_v6_obj.shape == (64, 168)
        self._test_hvm(hvm_it_v6_obj)

    def test_hvm_T(self):
        hvm_it_v6_obj = _load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"])).T
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
        hvm = _load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
        rdm_metric = RDMMetric()
        score = rdm_metric(hvm, hvm)
        assert score == 1.


class TestPCA(object):
    def test_noop(self):
        hvm = _load_hvm()
        pca = PCANeuroidCharacterization(max_components=1000)
        hvm_ = pca(hvm)
        xarray.testing.assert_equal(hvm, hvm_)

    def test_100(self):
        hvm = _load_hvm()
        pca = PCANeuroidCharacterization(max_components=100)
        hvm_ = pca(hvm)
        assert isinstance(hvm_, mkgu.assemblies.NeuroidAssembly)
        np.testing.assert_array_equal([hvm.shape[0], 100], hvm_.shape)


class TestNeuralFit(object):
    def test_equal(self):
        hvm = _load_hvm()
        neural_fit_metric = NeuralFitMetric()
        score = neural_fit_metric(hvm, hvm)
        assert score > 0.75


def _load_hvm(group=lambda hvm: hvm.multi_groupby(['obj', 'id'])):
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6 = group(hvm_it_v6)
    hvm_it_v6 = hvm_it_v6.mean(dim="presentation").squeeze("time_bin").T
    return hvm_it_v6
