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


def test_hvm_it_rdm():
    loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")

    hvm_it_v6_obj = _load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))

    assert hvm_it_v6_obj.shape == (64, 168)

    rsa_characterization = RSA()
    rsa = rsa_characterization(hvm_it_v6_obj)

    assert isinstance(rsa, mkgu.assemblies.DataAssembly)
    assert rsa.shape == (64, 64)
    assert rsa.values == approx(loaded, abs=1e-6)


def test_rdm_metric():
    hvm = _load_hvm(group=lambda hvm: hvm.multi_groupby(["category", "obj"]))
    rdm_metric = RDMMetric()
    score = rdm_metric(hvm, hvm)
    assert score == 1.


def test_pca_characterization_noop():
    hvm = _load_hvm()
    pca = PCANeuroidCharacterization(max_components=1000)
    hvm_ = pca(hvm)
    xarray.testing.assert_equal(hvm, hvm_)


def test_pca_characterization_100():
    hvm = _load_hvm()
    pca = PCANeuroidCharacterization(max_components=100)
    hvm_ = pca(hvm)
    assert isinstance(hvm_, mkgu.assemblies.NeuroidAssembly)
    np.testing.assert_array_equal([hvm.shape[0], 100], hvm_.shape)


def test_neural_fit_metric_nopca():
    hvm = _load_hvm()
    neural_fit_metric = NeuralFitMetric(pca_components=None)
    score = neural_fit_metric(hvm, hvm)
    assert 0.75 < score < 0.8


def test_neural_fit_metric_pca100():
    hvm = _load_hvm()
    neural_fit_metric = NeuralFitMetric(pca_components=100)
    score = neural_fit_metric(hvm, hvm)
    assert 0.10 < score < 0.15


def _load_hvm(group=lambda hvm: hvm.multi_groupby(['obj', 'id'])):
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6 = group(hvm_it_v6)
    hvm_it_v6 = hvm_it_v6.mean(dim="presentation").squeeze("time_bin").T
    return hvm_it_v6
