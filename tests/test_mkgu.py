#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mkgu
----------------------------------

Tests for `mkgu` module.
"""

import pytest
import numpy as np
from mkgu import assemblies
from mkgu import metrics
import pandas as pd
import xarray as xr
from pytest import approx

from mkgu import mkgu


@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/jjpr-mit/mkgu')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_nr_assembly_ctor():
    my_assembly = assemblies.NeuronRecordingAssembly(name="HvMWithDiscfade")


def test_load():
    print(os.getcwd())
    it_rdm = np.load("it_rdm.p", encoding="latin1")
    print(it_rdm)
    assert it_rdm.shape == (64, 64)


def test_hvm_it_rdm():
    loaded = np.load("it_rdm.p", encoding="latin1")
    assy_hvm = assemblies.NeuronRecordingAssembly(name="HvMWithDiscfade")
    rdm_hvm = metrics.RDM()
    bmk_hvm = metrics.Benchmark(rdm_hvm, assy_hvm)
    rdm = bmk_hvm.calculate()
    assert np.array_equal(rdm, loaded)


