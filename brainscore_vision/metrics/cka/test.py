import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import DataAssembly
from brainscore_vision import load_metric


def _make_data():
    rnd = RandomState(1)
    return DataAssembly(rnd.uniform(size=(300, 10)), coords={
        'stimulus_id': ('presentation', np.arange(300)), 'category': ('presentation', ['dummy'] * 300),
        'neuroid_id': ('neuroid', np.arange(10)), 'region': ('neuroid', ['Vdummy'] * 10)},
                        dims=['presentation', 'neuroid'])


def test_identity():
    cka = load_metric('cka')
    assembly = _make_data()
    similarity = cka(assembly, assembly)
    assert similarity == approx(1)


def test_crossvalidated_has_error():
    assembly = _make_data()
    ckacv = load_metric('cka_cv', crossvalidation_kwargs=dict(stratification_coord=None))
    similarity = ckacv(assembly, assembly)
    assert hasattr(similarity, 'error')
