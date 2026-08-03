import numpy as np
import pytest

from brainscore_vision import benchmark_registry, load_benchmark

REGIONS = ['V1', 'V2', 'V4', 'IT']
METRICS = ['ridgecv', 'pls']
IDENTIFIERS = [f'Li2026.{r}-{m}' for r in REGIONS for m in METRICS]


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_registered(identifier):
    assert identifier in benchmark_registry


@pytest.mark.parametrize('region', REGIONS)
def test_scored_leaf_is_ridgecv(region):
    # one scored leaf per region; pls stays runnable but out of the tree
    assert load_benchmark(f'Li2026.{region}-ridgecv').parent == region
    assert load_benchmark(f'Li2026.{region}-pls').parent is None


@pytest.mark.parametrize('region', REGIONS)
@pytest.mark.parametrize('metric', METRICS)
def test_benchmark_assembly(region, metric):
    benchmark = load_benchmark(f'Li2026.{region}-{metric}')
    assembly = benchmark._assembly
    # single region, window-reliability-filtered neuroids, full NSD stimulus set
    assert set(np.unique(assembly['region'].values)) == {region}
    assert assembly.sizes['presentation'] == 1000
    assert assembly.sizes['neuroid'] > 0
    # selection + ceiling are on the 70-170 ms window reliability (matches the scored response)
    assert (assembly['reliability_window'].values > 0.4).all()
    # paper-canonical best-window reliability retained as provenance; patch labels restored
    # (neuroid coords are MultiIndex levels, so check the index names, not assembly.coords)
    levels = assembly.indexes['neuroid'].names
    assert 'reliability' in levels
    assert 'arealabel' in levels
    assert len(benchmark._assembly.stimulus_set) == 1000


@pytest.mark.parametrize('region,expected_min', [('IT', 20000), ('V1', 2000), ('V2', 2200), ('V4', 3000)])
def test_reliable_neuroid_counts(region, expected_min):
    # window-matched (70-170 ms) reliable-unit counts: IT ~21.1k, V1 ~2.3k, V2 ~2.5k, V4 ~3.4k
    # (the paper's best-window magnitudes -- IT 26.7k -- live in the `reliability` coord)
    benchmark = load_benchmark(f'Li2026.{region}-ridgecv')
    assert benchmark._assembly.sizes['neuroid'] >= expected_min


@pytest.mark.parametrize('identifier', IDENTIFIERS)
def test_ceiling(identifier):
    benchmark = load_benchmark(identifier)
    ceiling = benchmark.ceiling
    assert 0 < float(ceiling) <= 1
    # DB-write contract: scalar center + error (-> BenchmarkInstance.ceiling_error) + per-neuroid raw
    assert ceiling.size == 1
    assert 'error' in ceiling.attrs and np.isfinite(ceiling.attrs['error']) and ceiling.attrs['error'] > 0
    assert ceiling.raw.dims == ('neuroid',)


def test_ridgecv_selects_alpha_via_dual_form():
    # alpha is fitted rather than pinned at 1, and the dual form keeps the
    # (n_targets, n_features) coefficient matrix off the peak at IT's ~21k units
    from brainscore_vision.metrics.regression_correlation.metric import DualRidgeCVRegression
    regression = load_benchmark('Li2026.V1-ridgecv')._similarity_metric.regression._regression
    assert isinstance(regression, DualRidgeCVRegression)
    assert len(regression.alphas) > 1


class TestITRDM:
    """RSA leaf. IT only: RSACeiling compares each subject against the mean of all,
    which needs several. V1/V2/V4 have 2 animals, where the leave-one-out bound
    collapses to a single A-vs-B correlation."""

    def test_registered_and_parented(self):
        assert 'Li2026.IT-rdm' in benchmark_registry
        assert load_benchmark('Li2026.IT-rdm').parent == 'IT'

    def test_only_it_has_an_rdm_leaf(self):
        for region in ['V1', 'V2', 'V4']:
            assert f'Li2026.{region}-rdm' not in benchmark_registry

    def test_subject_coord_exposed_for_rsa(self):
        # RSABenchmark groups by `subject`; Li2026 records the animal as `animal`
        assembly = load_benchmark('Li2026.IT-rdm')._assembly
        assert 'subject' in assembly.indexes['neuroid'].names
        assert (assembly['subject'].values == assembly['animal'].values).all()
        assert len(np.unique(assembly['subject'].values)) == 5

    def test_rdm_scores_the_same_units_as_the_encoding_leaf(self):
        rdm = load_benchmark('Li2026.IT-rdm')._assembly
        ridgecv = load_benchmark('Li2026.IT-ridgecv')._assembly
        assert rdm.sizes['neuroid'] == ridgecv.sizes['neuroid']
        assert rdm.sizes['presentation'] == 1000

    def test_ceiling_bounds(self):
        ceiling = load_benchmark('Li2026.IT-rdm').ceiling
        # Nili upper bound normalises the score; LOO lower bound is reported alongside
        assert 0 < float(ceiling) <= 1
        lower = ceiling.attrs['lower_bound_loo']
        assert 0 < lower < float(ceiling), 'LOO bound should sit below the upper bound'
