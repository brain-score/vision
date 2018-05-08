#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mkgu
----------------------------------

Tests for `mkgu` module.
"""

import os
import random

import numpy as np
import pytest
import xarray as xr
from pytest import approx

import mkgu
import mkgu.fetch
import mkgu.lookup
from mkgu import assemblies, fetch

_hvm_s3_url = "https://mkgu-dicarlolab-hvm.s3.amazonaws.com/hvm_neuronal_features.nc"


def test_nr_assembly_ctor():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    assert isinstance(assy_hvm, mkgu.assemblies.DataAssembly)


def test_load():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    assert assy_hvm.shape == (296, 268800, 1)
    print(assy_hvm)


def test_repr():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    repr_hvm = repr(assy_hvm)
    assert "neuroid" in repr_hvm
    assert "presentation" in repr_hvm
    assert "296" in repr_hvm
    assert "268800" in repr_hvm
    assert "animal" in repr_hvm
    print(repr_hvm)


def test_getitem():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    single = assy_hvm[0, 0, 0]
    assert type(single) is type(assy_hvm)
    assert single.values == approx(0.808021)


def test_lookup():
    assy = mkgu.assemblies.lookup_assembly("dicarlo.Majaj2015")
    assert assy.name == "dicarlo.Majaj2015"
    store = assy.assembly_store_maps[0]
    assert store.role == "dicarlo.Majaj2015"
    assert store.assembly_store_model.location_type == "S3"
    assert store.assembly_store_model.location == _hvm_s3_url


def test_lookup_bad_name():
    with pytest.raises(mkgu.assemblies.AssemblyLookupError) as err:
        mkgu.assemblies.lookup_assembly("BadName")


def test_fetch():
    assy_record = mkgu.assemblies.lookup_assembly("dicarlo.Majaj2015")
    local_paths = fetch.fetch_assembly(assy_record)
    assert len(local_paths) == 1
    print(local_paths["dicarlo.Majaj2015"])
    assert os.path.exists(local_paths["dicarlo.Majaj2015"])


def test_wrap():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    hvm_v6 = assy_hvm.sel(variation=6)
    assert isinstance(hvm_v6, assemblies.NeuronRecordingAssembly)

    hvm_it_v6 = hvm_v6.sel(region="IT")
    assert isinstance(hvm_it_v6, assemblies.NeuronRecordingAssembly)

    hvm_it_v6.coords["cat_obj"] = hvm_it_v6.coords["category_name"] + hvm_it_v6.coords["object_name"]
    hvm_it_v6.load()
    hvm_it_v6_grp = hvm_it_v6.multi_groupby(["category_name", "object_name"])
    assert not isinstance(hvm_it_v6_grp, xr.core.groupby.GroupBy)
    assert isinstance(hvm_it_v6_grp, assemblies.GroupbyBridge)

    hvm_it_v6_obj = hvm_it_v6_grp.mean(dim="presentation")
    assert isinstance(hvm_it_v6_obj, assemblies.NeuronRecordingAssembly)

    hvm_it_v6_sqz = hvm_it_v6_obj.squeeze("time_bin")
    assert isinstance(hvm_it_v6_sqz, assemblies.NeuronRecordingAssembly)

    hvm_it_v6_t = hvm_it_v6_sqz.T
    assert isinstance(hvm_it_v6_t, assemblies.NeuronRecordingAssembly)


def test_multi_group():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    hvm_it_v6 = assy_hvm.sel(variation=6).sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6_obj = hvm_it_v6.multi_groupby(["category_name", "object_name"]).mean(dim="presentation")
    assert "category_name" in hvm_it_v6_obj.indexes["presentation"].names
    assert "object_name" in hvm_it_v6_obj.indexes["presentation"].names


def test_get_stimulus_set():
    stimulus_set = mkgu.get_stimulus_set("dicarlo.hvm")
    assert "hash_id" in stimulus_set.columns
    assert stimulus_set.shape == (5760, 17)
    assert os.path.exists(stimulus_set.image_paths[random.choice(stimulus_set["hash_id"])])
    for hash_id in stimulus_set['hash_id']:
        image_path = stimulus_set.get_image(hash_id)
        assert os.path.exists(image_path)


def test_stimulus_set_from_assembly():
    assy_hvm = mkgu.get_assembly(name="dicarlo.Majaj2015")
    stimulus_set = assy_hvm.attrs["stimulus_set"]
    assert stimulus_set.shape[0] == np.unique(assy_hvm["image_id"]).shape[0]
    for hash_id in stimulus_set['hash_id']:
        image_path = stimulus_set.get_image(hash_id)
        assert os.path.exists(image_path)
