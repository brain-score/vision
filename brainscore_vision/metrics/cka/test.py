import numpy as np
from numpy.random import RandomState
from pytest import approx

from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_vision import load_metric


def _make_data(random_seed=1, n=300, m=10):
    rnd = RandomState(random_seed)
    return DataAssembly(rnd.uniform(size=(n, m)), coords={
        'stimulus_id': ('presentation', np.arange(n)), 'category': ('presentation', ['dummy'] * n),
        'neuroid_id': ('neuroid', np.arange(m)), 'region': ('neuroid', ['Vdummy'] * m)},
                        dims=['presentation', 'neuroid'])


def test_identity():
    cka = load_metric('cka')
    assembly = _make_data()
    similarity = cka(assembly, assembly)
    assert similarity == approx(1)


def test_large_data():
    cka_unbiased = load_metric('cka')
    cka_biased = load_metric('cka_biased')
    assembly1 = _make_data(random_seed=1, n=1000, m=10_000)
    assembly2 = _make_data(random_seed=2, n=1000, m=10_000)
    
    similarity_unbiased = cka_unbiased(assembly1, assembly2)
    assert np.abs(similarity_unbiased) < 0.1 # Unbiased CKA should be close to 0 for independent data

    similarity_biased = cka_biased(assembly1, assembly2)
    assert similarity_biased > 0.9 # Biased CKA should be large especially with large feature dimension

def test_crossvalidated_has_error():
    assembly = _make_data()
    ckacv = load_metric('cka_cv', crossvalidation_kwargs=dict(stratification_coord=None))
    similarity = ckacv(assembly, assembly)
    assert hasattr(similarity, 'error')
