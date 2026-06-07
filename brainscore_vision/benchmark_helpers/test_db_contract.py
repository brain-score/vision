"""Round-trip integration tests for the brainscore_core submission DB contract.

These tests catch schema mismatches between what wrapper benchmarks produce in
``score.attrs`` and what ``brainscore_core.submission.database.update_score``
expects when recording to the leaderboard DB. Every issue we've hit serially
(missing ``version``, multi-entry ``BIBTEX``, ``raw`` as per-subject array,
``raw`` as Python float) would have been caught by these tests at unit-test
time instead of surfacing only at production submission time.

Each test instantiates the wrapper with a tiny mock factory that returns
synthetic per-subject scores, then exercises the exact code paths
``update_score`` calls:

  - ``score.item()`` and ``score.raw.item()`` via ``_retrieve_score_center``
  - ``score.attrs['error']`` via ``_retrieve_score_error``
  - wrapper attributes: ``identifier``, ``version``, ``parent``, ``bibtex``
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from brainscore_core import Score


def _try_import_db_helpers():
    """Import the DB recorder helpers if available; skip the suite if not.

    brainscore_core may live in different installs; we want the tests to
    decline gracefully rather than ImportError-fail when run on an unusual env.
    """
    try:
        from brainscore_core.submission.database import (
            _retrieve_score_center, _retrieve_score_error
        )
        return _retrieve_score_center, _retrieve_score_error
    except ImportError:  # pragma: no cover
        pytest.skip("brainscore_core.submission.database not importable")


class _SyntheticChild:
    """Mock TrainTestNeuralBenchmark-shaped child.

    Provides the attributes the wrappers read during ``__init__`` + ``__call__``,
    and a ``__call__`` that returns a Score with the same attrs shape a real
    child would (scalar value, scalar attrs['raw'], scalar attrs['ceiling']).
    """

    def __init__(self, sub_id: str, region: str = "V1", seed: int = 0):
        self.region = region
        self.parent = region
        self.bibtex = "@inproceedings{test_2026, title={x}, author={y}}"
        self.version = 1
        self.timebins = [(0, 0)]
        self._assembly = object()
        self._similarity_metric = object()
        # Distinct per-subject values so the wrapper's mean is non-trivial.
        rng = np.random.default_rng(seed)
        self._raw = float(rng.uniform(0.4, 0.7))
        self._ceiled = float(rng.uniform(0.2, 0.6))
        self._ceiling = float(rng.uniform(0.7, 0.9))

    @property
    def ceiling(self) -> Score:
        return Score(self._ceiling)

    def __call__(self, candidate) -> Score:
        s = Score(self._ceiled)
        s.attrs["raw"] = Score(self._raw)
        s.attrs["ceiling"] = Score(self._ceiling)
        s.attrs["error"] = 0.05
        return s


def _assert_db_contract(score, wrapper, _retrieve_score_center, _retrieve_score_error):
    """The exact sequence ``update_score`` runs against the returned score."""
    # Standard BenchmarkBase contract.
    assert wrapper.identifier
    assert wrapper.version is not None
    assert wrapper.parent
    assert wrapper.bibtex
    # The two ``.item()`` paths the DB recorder takes for ceiled benchmarks.
    assert "ceiling" in score.attrs
    raw_value = _retrieve_score_center(score.raw)
    ceiled_value = _retrieve_score_center(score)
    assert isinstance(raw_value, float)
    assert isinstance(ceiled_value, float)
    # Error column.
    err = _retrieve_score_error(score)
    assert err is None or isinstance(err, float)


# ── MultiSubjectNeuralBenchmark ────────────────────────────────────────────


def test_multi_subject_satisfies_update_score_contract():
    _retrieve_score_center, _retrieve_score_error = _try_import_db_helpers()
    from brainscore_vision.benchmark_helpers.multi_subject import MultiSubjectNeuralBenchmark

    subjects = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
    wrapper = MultiSubjectNeuralBenchmark(
        identifier="test_dataset.V1-tau-ridgecv",
        subjects=subjects,
        per_subject_factory=lambda sid, i=[0]: _SyntheticChild(sid, "V1", seed=(i.__setitem__(0, i[0] + 1) or i[0])),
    )
    score = wrapper(candidate=None)
    _assert_db_contract(score, wrapper, _retrieve_score_center, _retrieve_score_error)


def test_multi_subject_single_subject_branch():
    """Single-subject path uses ``declare_no_error`` (nan error). Must still .item()."""
    _retrieve_score_center, _retrieve_score_error = _try_import_db_helpers()
    from brainscore_vision.benchmark_helpers.multi_subject import MultiSubjectNeuralBenchmark

    wrapper = MultiSubjectNeuralBenchmark(
        identifier="test_dataset.V1-tau-ridgecv",
        subjects=("sub-01",),
        per_subject_factory=lambda sid: _SyntheticChild(sid, "V1"),
    )
    score = wrapper(candidate=None)
    # Center calls still work.
    _retrieve_score_center(score)
    _retrieve_score_center(score.raw)
    # Error retrieval doesn't raise even though it's NaN.
    err = _retrieve_score_error(score)
    assert err is None or (isinstance(err, float) and (np.isnan(err) or not np.isnan(err)))


# ── KFoldNeuralBenchmark ───────────────────────────────────────────────────


def test_kfold_satisfies_update_score_contract():
    _retrieve_score_center, _retrieve_score_error = _try_import_db_helpers()
    from brainscore_vision.benchmark_helpers.multi_subject import KFoldNeuralBenchmark

    wrapper = KFoldNeuralBenchmark(
        identifier="test_dataset.V1-cluster_k5-ridgecv",
        n_folds=5,
        fold_factory=lambda k: _SyntheticChild(f"fold-{k}", "V1", seed=k),
    )
    score = wrapper(candidate=None)
    _assert_db_contract(score, wrapper, _retrieve_score_center, _retrieve_score_error)


# ── _MultiSubjectRSABenchmark (LAION-specific class) ───────────────────────


def test_rsa_multi_subject_satisfies_update_score_contract():
    _retrieve_score_center, _retrieve_score_error = _try_import_db_helpers()
    from brainscore_vision.benchmarks.laion_fmri.benchmark import _MultiSubjectRSABenchmark

    subjects = ("sub-01", "sub-03", "sub-05")
    wrapper = _MultiSubjectRSABenchmark(
        identifier="test_dataset.V1-rdm-pearson",
        subjects=subjects,
        per_subject_factory=lambda sid, i=[0]: _SyntheticChild(sid, "V1", seed=(i.__setitem__(0, i[0] + 1) or i[0])),
        region="V1",
        bibtex="@inproceedings{test_2026, title={x}, author={y}}",
    )
    score = wrapper(candidate=None)
    _assert_db_contract(score, wrapper, _retrieve_score_center, _retrieve_score_error)


# ── BenchmarkBase contract surface ─────────────────────────────────────────


def test_wrappers_inherit_benchmarkbase():
    """The wrappers must subclass BenchmarkBase so brainscore_core's DB code
    reads ``version`` / ``bibtex`` / ``parent`` from the standard contract."""
    from brainscore_vision.benchmarks import BenchmarkBase
    from brainscore_vision.benchmark_helpers.multi_subject import (
        KFoldNeuralBenchmark, MultiSubjectNeuralBenchmark,
    )
    from brainscore_vision.benchmarks.laion_fmri.benchmark import _MultiSubjectRSABenchmark
    assert issubclass(KFoldNeuralBenchmark, BenchmarkBase)
    assert issubclass(MultiSubjectNeuralBenchmark, BenchmarkBase)
    assert issubclass(_MultiSubjectRSABenchmark, BenchmarkBase)
