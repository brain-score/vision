import numpy as np
import pytest

from brainscore_vision import benchmark_registry, load_benchmark

REGIONS = ['V1', 'V2', 'V4', 'IT']
METRICS = ['pls', 'ridge']
IDENTIFIERS = [f'Li2026.{r}-{m}' for r in REGIONS for m in METRICS]


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_registered(identifier):
    assert identifier in benchmark_registry


@pytest.mark.parametrize('region', REGIONS)
@pytest.mark.parametrize('metric', METRICS)
def test_benchmark_assembly(region, metric):
    benchmark = load_benchmark(f'Li2026.{region}-{metric}')
    assembly = benchmark._assembly
    # single region, reliability-filtered neuroids, full NSD stimulus set
    assert set(np.unique(assembly['region'].values)) == {region}
    assert assembly.sizes['presentation'] == 1000
    assert assembly.sizes['neuroid'] > 0
    assert (assembly['reliability'].values > 0.4).all()
    assert len(benchmark._assembly.stimulus_set) == 1000


@pytest.mark.parametrize('region,expected_min', [('IT', 20000), ('V1', 1500), ('V2', 1500), ('V4', 2000)])
def test_reliable_neuroid_counts(region, expected_min):
    # reproduces the paper's reliable-unit magnitudes (IT ~26.7k; EVC few thousand each)
    benchmark = load_benchmark(f'Li2026.{region}-pls')
    assert benchmark._assembly.sizes['neuroid'] >= expected_min


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_ceiling(identifier):
    benchmark = load_benchmark(identifier)
    ceiling = benchmark.ceiling
    assert 0 < float(ceiling) <= 1
