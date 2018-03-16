#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mkgu
----------------------------------

Tests for `mkgu` module.
"""

import os

import numpy as np
from pytest import approx

import mkgu
from mkgu.metrics.rdm import RSA, RDMMetric


def test_hvm_it_rdm():
    loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")

    hvm_it_v6_obj = _load_hvm()

    assert hvm_it_v6_obj.shape == (64, 168)

    rsa_characterization = RSA()
    rsa = rsa_characterization.apply(hvm_it_v6_obj)

    assert rsa.shape == (64, 64)
    assert rsa == approx(loaded, abs=1e-6)


def test_rdm_metric():
    hvm = _load_hvm()
    rdm_metric = RDMMetric()
    score = rdm_metric.apply(hvm, hvm)
    assert score == 1.


def _load_hvm():
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6_obj = hvm_it_v6.multi_groupby(["category", "obj"]).mean(dim="presentation").squeeze("time_bin").T
    return hvm_it_v6_obj
