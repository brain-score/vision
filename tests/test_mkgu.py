#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mkgu
----------------------------------

Tests for `mkgu` module.
"""

import os
import pytest
import numpy as np
import mkgu
from mkgu import assemblies, fetch
from mkgu import metrics
import pandas as pd
import xarray as xr
from pytest import approx

_hvm_s3_url = "https://mkgu-dicarlolab-hvm.s3.amazonaws.com/hvm_neuronal_features.nc"


@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/dicarlolab/mkgu')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_nr_assembly_ctor():
    assy_hvm = mkgu.get_assembly(name="HvM")


def test_np_load():
    print(os.getcwd())
    it_rdm = np.load("it_rdm.p", encoding="latin1")
    print(it_rdm)
    assert it_rdm.shape == (64, 64)


def test_hvm_it_rdm():
    loaded = np.load(os.path.join(os.path.dirname(__file__), "it_rdm.p"), encoding="latin1")

    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.coords["cat_obj"] = hvm_it_v6.coords["category"] + hvm_it_v6.coords["obj"]
    hvm_it_v6.load()
    hvm_it_v6_obj = hvm_it_v6.groupby("cat_obj").mean(dim="presentation").squeeze("time_bin").T

    assert hvm_it_v6_obj.shape == (64, 168)

    rdm_hvm = metrics.RDM()
    bmk_hvm = metrics.Benchmark(rdm_hvm, hvm_it_v6_obj)
    rdm = bmk_hvm.calculate()

    assert rdm.shape == (64, 64)
    assert rdm == approx(loaded, abs=1e-6)


def test_load():
    assy_hvm = mkgu.get_assembly(name="HvM")
    assert assy_hvm.shape == (296, 268800, 1)
    print(assy_hvm)


def test_lookup():
    assy = fetch.get_lookup().lookup_assembly("HvM")
    assert assy.name == "HvM"
    store = list(assy.stores.values())[0]
    assert store.role == "HvM"
    assert store.store.type == "S3"
    assert store.store.location == _hvm_s3_url


def test_lookup_bad_name():
    with pytest.raises(mkgu.fetch.AssemblyLookupError) as err:
        fetch.get_lookup().lookup_assembly("BadName")


def test_fetch():
    assy_record = fetch.get_lookup().lookup_assembly("HvM")
    local_paths = fetch.fetch_assembly(assy_record)
    assert len(local_paths) == 1
    print(local_paths["HvM"])
    assert os.path.exists(local_paths["HvM"])



