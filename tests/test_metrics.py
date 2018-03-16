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
from mkgu.metrics.rdm import RSA


def test_hvm_it_rdm():
    loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")

    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6_obj = hvm_it_v6.multi_groupby(["category", "obj"]).mean(dim="presentation").squeeze("time_bin").T

    assert hvm_it_v6_obj.shape == (64, 168)

    rsa_characterization = RSA()
    rsa = rsa_characterization.apply(hvm_it_v6_obj)

    assert rsa.shape == (64, 64)
    assert rsa == approx(loaded, abs=1e-6)
