"""Tests for multi-subject wrapper preallocate_memory + RDM suffix detection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brainscore_vision.benchmark_helpers.memory import (
    MemoryEstimate,
    _is_rdm_benchmark,
)


def _fake_estimate(total_gb: float, available_gb: float) -> MemoryEstimate:
    return MemoryEstimate(
        num_stimuli=100, num_trials=1, num_features=1024, num_timebins=1,
        activation_gb=total_gb / 6, total_estimated_gb=total_gb,
        available_gb=available_gb, formula_type="fallback",
    )


# ---------------------------------------------------------------------------
# RDM suffix detection
# ---------------------------------------------------------------------------

class TestIsRdmBenchmark:
    @pytest.mark.parametrize("ident", [
        "Allen2022_fmri.IT-rdm",
        "Zerbe2026_fmri.V1-rdm-pearson",
        "Zerbe2026_fmri.IT-rdm-spearman",
        "anything-rdm-anything",
    ])
    def test_matches_rdm_anywhere(self, ident):
        b = MagicMock()
        b.identifier = ident
        with patch("brainscore_vision.benchmark_helpers.memory.isinstance", lambda *_: False):
            assert _is_rdm_benchmark(b) is True

    @pytest.mark.parametrize("ident", [
        "MajajHong2015.IT-pls",
        "Allen2022_fmri.V4-ridge",
        "Gifford2022.IT-ridgecv",
    ])
    def test_does_not_match_non_rdm(self, ident):
        b = MagicMock()
        b.identifier = ident
        with patch("brainscore_vision.benchmark_helpers.memory.isinstance", lambda *_: False):
            assert _is_rdm_benchmark(b) is False


# ---------------------------------------------------------------------------
# Multi-subject wrapper preallocate_memory
# ---------------------------------------------------------------------------

class TestScaledPreallocateMemory:
    def _wrapper_class(self):
        from brainscore_vision.benchmark_helpers.multi_subject import MultiSubjectNeuralBenchmark
        return MultiSubjectNeuralBenchmark

    def _build_wrapper(self, peak_aggregation: float):
        WrapperCls = self._wrapper_class()
        child = MagicMock()
        child.region = "IT"
        child.bibtex = "@x{x}"
        child.version = 1
        child.timebins = [(0, 0)]
        wrapper = object.__new__(WrapperCls)
        wrapper.identifier = "Test.IT-tau-ridgecv"
        wrapper._subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
        wrapper._factory = lambda sub_id: child
        wrapper._peak_aggregation = float(peak_aggregation)
        wrapper.region = child.region
        wrapper.parent = child.region
        wrapper.bibtex = child.bibtex
        wrapper.version = child.version
        wrapper.timebins = child.timebins
        return wrapper

    def test_aggregation_scales_estimate(self):
        wrapper = self._build_wrapper(peak_aggregation=5.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=_fake_estimate(total_gb=20.0, available_gb=200.0),
        ) as mock_probe:
            wrapper.preallocate_memory(candidate)
        mock_probe.assert_called_once()
        args, kwargs = mock_probe.call_args
        assert kwargs.get("raise_if_oom") is False

    def test_raises_when_scaled_estimate_exceeds_available(self):
        wrapper = self._build_wrapper(peak_aggregation=5.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=_fake_estimate(total_gb=20.0, available_gb=50.0),
        ):
            with pytest.raises(MemoryError, match=r"100\.0 GB.*20\.0.*5\.0"):
                wrapper.preallocate_memory(candidate)

    def test_does_not_raise_when_scaled_estimate_fits(self):
        wrapper = self._build_wrapper(peak_aggregation=5.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=_fake_estimate(total_gb=20.0, available_gb=200.0),
        ):
            estimate = wrapper.preallocate_memory(candidate)
        assert estimate is not None
        # Returned estimate is the unscaled per-child estimate, not the scaled total.
        assert estimate.total_estimated_gb == pytest.approx(20.0)

    def test_aggregation_default_is_one(self):
        wrapper = self._build_wrapper(peak_aggregation=1.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=_fake_estimate(total_gb=10.0, available_gb=15.0),
        ):
            wrapper.preallocate_memory(candidate)  # 10 × 1.0 < 15 → no raise

    def test_returns_none_when_probe_skipped(self):
        wrapper = self._build_wrapper(peak_aggregation=5.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=None,
        ):
            assert wrapper.preallocate_memory(candidate) is None

    def test_raise_if_oom_false_suppresses_exception(self):
        wrapper = self._build_wrapper(peak_aggregation=5.0)
        candidate = MagicMock()
        with patch(
            "brainscore_vision.benchmark_helpers.memory.preallocate_memory",
            return_value=_fake_estimate(total_gb=20.0, available_gb=50.0),
        ):
            estimate = wrapper.preallocate_memory(candidate, raise_if_oom=False)
        assert estimate is not None
        assert estimate.total_estimated_gb == pytest.approx(20.0)


class TestKFoldPreallocateMemory(TestScaledPreallocateMemory):
    def _wrapper_class(self):
        from brainscore_vision.benchmark_helpers.multi_subject import KFoldNeuralBenchmark
        return KFoldNeuralBenchmark

    def _build_wrapper(self, peak_aggregation: float):
        WrapperCls = self._wrapper_class()
        child = MagicMock()
        child.region = "IT"
        child.bibtex = "@x{x}"
        child.version = 1
        wrapper = object.__new__(WrapperCls)
        wrapper.identifier = "Test.IT-cluster_k5-ridgecv"
        wrapper._n_folds = 5
        wrapper._factory = lambda k: child
        wrapper._peak_aggregation = float(peak_aggregation)
        wrapper.region = child.region
        wrapper.parent = child.region
        wrapper.bibtex = child.bibtex
        wrapper.version = child.version
        return wrapper


# ---------------------------------------------------------------------------
# LAION-fMRI per-pool aggregation factor
# ---------------------------------------------------------------------------

class TestPeakAggregationForPool:
    def test_persubject_pool_returns_n_subjects(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import (
            _peak_aggregation_for_pool, DEFAULT_SUBJECTS,
        )
        assert _peak_aggregation_for_pool(
            "Zerbe2026_fmri_persubject", DEFAULT_SUBJECTS
        ) == float(len(DEFAULT_SUBJECTS))

    def test_shared_pool_returns_one(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import (
            _peak_aggregation_for_pool, DEFAULT_SUBJECTS,
        )
        assert _peak_aggregation_for_pool("Zerbe2026_fmri", DEFAULT_SUBJECTS) == 1.0

    def test_persubject_scales_with_subject_count(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import _peak_aggregation_for_pool
        assert _peak_aggregation_for_pool("Zerbe2026_fmri_persubject", ["sub-01"]) == 1.0
        assert _peak_aggregation_for_pool(
            "Zerbe2026_fmri_persubject", ["sub-01", "sub-02"]
        ) == 2.0
