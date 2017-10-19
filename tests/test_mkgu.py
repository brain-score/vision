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


def test_nr_assembly_ctor():
    assy_hvm = mkgu.get_assembly(name="HvM")
    assert isinstance(assy_hvm, mkgu.assemblies.DataAssembly)


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


def test_repr():
    assy_hvm = mkgu.get_assembly(name="HvM")
    repr_hvm = repr(assy_hvm)
    assert "neuroid" in repr_hvm
    assert "presentation" in repr_hvm
    assert "296" in repr_hvm
    assert "268800" in repr_hvm
    assert "animal" in repr_hvm
    print(repr_hvm)


def test_getitem():
    assy_hvm = mkgu.get_assembly(name="HvM")
    single = assy_hvm[0, 0, 0]
    assert type(single) is type(assy_hvm)
    assert single.values == approx(0.808021)


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


def test_wrap():
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_v6 = assy_hvm.sel(var="V6")
    assert isinstance(hvm_v6, assemblies.NeuronRecordingAssembly)

    hvm_it_v6 = hvm_v6.sel(region="IT")
    assert isinstance(hvm_it_v6, assemblies.NeuronRecordingAssembly)

    hvm_it_v6.coords["cat_obj"] = hvm_it_v6.coords["category"] + hvm_it_v6.coords["obj"]
    hvm_it_v6.load()
    hvm_it_v6_grp = hvm_it_v6.multi_groupby(["category", "obj"])
    assert not isinstance(hvm_it_v6_grp, xr.core.groupby.GroupBy)
    assert isinstance(hvm_it_v6_grp, assemblies.GroupbyBridge)

    hvm_it_v6_obj = hvm_it_v6_grp.mean(dim="presentation")
    assert isinstance(hvm_it_v6_obj, assemblies.NeuronRecordingAssembly)

    hvm_it_v6_sqz = hvm_it_v6_obj.squeeze("time_bin")
    assert isinstance(hvm_it_v6_sqz, assemblies.NeuronRecordingAssembly)

    hvm_it_v6_t = hvm_it_v6_sqz.T
    assert isinstance(hvm_it_v6_t, assemblies.NeuronRecordingAssembly)


def test_multi_group():
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6_obj = hvm_it_v6.multi_groupby(["category", "obj"]).mean(dim="presentation")
    assert "category" in hvm_it_v6_obj.indexes["presentation"].names
    assert "obj" in hvm_it_v6_obj.indexes["presentation"].names





