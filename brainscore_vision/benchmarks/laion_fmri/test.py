"""Tests for the LAION-fMRI benchmark registry and basic load paths.

The headline registry is 20 variants — 8 shared-pool ridge, 8 persubject-pool
ridge, and 4 shared-pool RSA. Non-headline variants (cluster_k5, per-OOD-category,
IT_full) live as factory functions and are exercised separately in
`usage_examples.ipynb`; they're not registered for the leaderboard so the
registry tests don't cover them.

All scoring tests require both the LAION-fMRI assemblies (Brain-Score S3, CC0)
and the stimulus DUA-gated images (locally cached via `laion-fmri request-access`).
They're marked `@pytest.mark.private_access` per Brain-Score's standard convention
for benchmarks requiring restricted resources.
"""

import pytest

from brainscore_vision import load_benchmark


# ── Lean headline identifiers ─────────────────────────────────────────────
# Kept as explicit lists (not generators over REGIONS / SIMPLE_SPLITS) so
# this file's expected set is independent of the constants in benchmark.py —
# if someone adds a new split or region constant, these tests still describe
# the registered surface area.

_RIDGE_REGIONS = ("V1", "V2", "V4", "IT")
_RIDGE_SPLITS = ("tau", "ood")
_RIDGE_FAMILIES = ("Zerbe2026_fmri", "Zerbe2026_fmri_persubject")

_RIDGE_VARIANTS = [
    f"{fam}.{region}-{split}-ridgecv"
    for fam in _RIDGE_FAMILIES
    for region in _RIDGE_REGIONS
    for split in _RIDGE_SPLITS
]  # 2 * 4 * 2 = 16

_RSA_VARIANTS = [
    f"Zerbe2026_fmri.{region}-rdm-pearson" for region in _RIDGE_REGIONS
]  # 4 (shared pool only — Nili ceiling requires shared stim across subjects)

_ALL_HEADLINE_VARIANTS = _RIDGE_VARIANTS + _RSA_VARIANTS  # 20


class TestRegistry:
    """The lean registry exposes exactly the 20 headline variants and nothing else."""

    def test_variant_count(self):
        assert len(_ALL_HEADLINE_VARIANTS) == 20, (
            f"Expected 20 headline variants, got {len(_ALL_HEADLINE_VARIANTS)}. "
            f"If you added/removed variants in __init__.py, update the lists at "
            f"the top of this file too."
        )

    def test_registry_matches_expected(self):
        """The registry's Zerbe2026_fmri* entries exactly equal the headline set."""
        from brainscore_vision import benchmark_registry
        import brainscore_vision.benchmarks.laion_fmri  # populate

        registered = {k for k in benchmark_registry if k.startswith("Zerbe2026_fmri")}
        expected = set(_ALL_HEADLINE_VARIANTS)
        missing = expected - registered
        extra = registered - expected
        assert not missing, f"Missing from registry: {sorted(missing)}"
        assert not extra, f"Unexpected in registry: {sorted(extra)}"

    @pytest.mark.private_access
    @pytest.mark.parametrize("identifier", _ALL_HEADLINE_VARIANTS)
    def test_load(self, identifier):
        """Each registered factory instantiates without error."""
        benchmark = load_benchmark(identifier)
        assert benchmark is not None
        assert benchmark.identifier == identifier


class TestSplitResolution:
    """Pure split-resolution logic — no data required."""

    @pytest.mark.parametrize(
        "split, expected_canonical, expected_filter",
        [
            ("tau", "tau", None),
            ("ood", "ood", None),
            ("ood_shape", "ood", ("shape",)),
            ("ood_illusion-classic", "ood", ("illusion-classic",)),
            ("cluster_k5_0", "cluster_k5_0", None),
        ],
    )
    def test_resolve_split(self, split, expected_canonical, expected_filter):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import _resolve_split

        canonical, ood_filter = _resolve_split(split)
        assert canonical == expected_canonical
        assert ood_filter == expected_filter


class TestNonHeadlineFactoriesConstruct:
    """Non-headline factories still need to construct cleanly even though they're
    intentionally absent from the registry. Exercised so a refactor that breaks
    them is caught even if no leaderboard variant uses them."""

    @pytest.mark.private_access
    def test_cluster_cv_constructs(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRIClusterCV

        b = LAIONfMRIClusterCV("V4")
        assert b.identifier == "Zerbe2026_fmri.V4-cluster_k5-ridgecv"

    @pytest.mark.private_access
    def test_per_ood_category_constructs(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRI

        b = LAIONfMRI("IT", "ood_gabor")
        assert "ood_gabor" in b.identifier

    @pytest.mark.private_access
    def test_it_full_constructs(self):
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRI

        b = LAIONfMRI("IT_full", "tau")
        assert "IT_full" in b.identifier

    def test_persubject_rsa_rejected(self):
        """RSA on persubject pool should raise — Nili ceiling requires shared stim."""
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRIRSA

        with pytest.raises(ValueError, match="shared pool"):
            LAIONfMRIRSA("IT", dataset_prefix="Zerbe2026_fmri_persubject")


class TestAlexNetSmoke:
    """End-to-end scoring smoke tests with alexnet covering each benchmark family.

    Asserts the score path completes and returns a sane (finite, ≤ 1.5) value
    against the cached baselines from the May 2026 sweep. Each cell takes
    ~20-30s with cached activations, ~10min for a cold start.
    """

    @pytest.mark.private_access
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "identifier",
        [
            "Zerbe2026_fmri.V1-tau-ridgecv",                   # shared ridgecv
            "Zerbe2026_fmri_persubject.IT-tau-ridgecv",        # persubject ridgecv
            "Zerbe2026_fmri.IT-rdm-pearson",                   # shared RSA
        ],
    )
    def test_model_runs(self, identifier):
        from brainscore_vision import load_model

        benchmark = load_benchmark(identifier)
        model = load_model("alexnet")
        score = benchmark(model)
        assert score is not None
        value = float(score.values)
        assert value == value, f"non-finite score: {value}"  # NaN check
        assert -1.0 <= value <= 1.5, f"score out of expected range: {value}"


class TestUncertaintyContract:
    """Each benchmark wrapper must return a Score satisfying the REPORTING_UNCERTAINTY
    spec: finite ``error`` on the ceiled scale (or nan with a declared reason),
    plus disaggregated ``raw`` at the finest resampleable grain.
    """

    @pytest.mark.private_access
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "identifier",
        [
            "Zerbe2026_fmri.V1-tau-ridgecv",     # MultiSubjectNeuralBenchmark wrapper
            "Zerbe2026_fmri.V1-rdm-pearson",     # _MultiSubjectRSABenchmark wrapper
        ],
    )
    def test_multisubject_wrapper_uncertainty(self, identifier):
        from brainscore_vision import load_model
        from brainscore_vision.benchmark_helpers.validate_error import (
            assert_valid_uncertainty,
        )

        benchmark = load_benchmark(identifier)
        score = benchmark(load_model("alexnet"))
        assert_valid_uncertainty(score)
        assert score.attrs["error_over"] == ["subject"]
        assert score.attrs["n_bootstrap"] == 200

    @pytest.mark.private_access
    @pytest.mark.slow
    def test_kfold_wrapper_uncertainty(self):
        from brainscore_vision import load_model
        from brainscore_vision.benchmark_helpers.validate_error import (
            assert_valid_uncertainty,
        )
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRIClusterCV

        benchmark = LAIONfMRIClusterCV("V1")
        score = benchmark(load_model("alexnet"))
        assert_valid_uncertainty(score)
        assert score.attrs["error_over"] == ["fold"]
        assert score.attrs["n_bootstrap"] == 200

    @pytest.mark.private_access
    @pytest.mark.slow
    def test_single_subject_declares_no_error(self):
        from brainscore_vision import load_model
        from brainscore_vision.benchmark_helpers.validate_error import (
            assert_valid_uncertainty,
        )
        from brainscore_vision.benchmarks.laion_fmri.benchmark import LAIONfMRI

        benchmark = LAIONfMRI("V1", "tau", subjects=("sub-01",))
        score = benchmark(load_model("alexnet"))
        assert_valid_uncertainty(score)
        import math
        assert math.isnan(float(score.attrs["error"]))
        assert score.attrs.get("error_nan_reason")
