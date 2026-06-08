"""Aggregator wrappers + assembly utilities for multi-subject / k-fold benchmarks.

These helpers came out of the LAION-fMRI plugin (introduced 2026-05) but are
intentionally written without any LAION-specific assumptions so any benchmark
with the same shape — multiple subjects with disjoint voxels, or k-fold CV
over a single TrainTestNeuralBenchmark — can reuse them.

Three pieces live here:

- :class:`MultiSubjectNeuralBenchmark`: runs one child benchmark per subject,
  aggregates ceiled scores with mean across subjects and preserves per-subject
  detail in ``score.attrs["raw_subjects"]`` + ``score.attrs[<subject_id>]``.
- :class:`KFoldNeuralBenchmark`: averages across an arbitrary set of fold
  benchmarks (e.g. CLIP-cluster CV), preserving per-fold detail in
  ``score.attrs["raw_folds"]``.
- :func:`block_diagonal_concat`: stitch per-subject assembly slices into a
  single (presentation, neuroid) array where each subject occupies a diagonal
  block and off-diagonal cells are NaN. Useful when subjects have disjoint
  neuroids but share a presentation index.

Neither aggregator class subclasses :class:`~brainscore_vision.benchmarks.BenchmarkBase`
(yet) because Brain-Score's invocation path only requires ``identifier`` and a
``__call__(candidate) -> Score`` — promoting to a full BenchmarkBase can happen
once a second caller appears that needs the extra metadata.
"""

from __future__ import annotations

import gc
from typing import Callable, Sequence

import numpy as np
import xarray as xr

from brainscore_core import Score
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark
from brainscore_vision.metric_helpers.bootstrap_error import (
    attach_error,
    declare_no_error,
)


def _release(child) -> None:
    """Best-effort release of a benchmark child's heavy state before GC.

    Children hold a several-hundred-MB neuroid assembly via ``_assembly`` and
    cached BenchmarkBase ceiling closures that capture it. Nulling explicitly
    breaks the closure cycle so refcount drops to zero without waiting for the
    cycle collector.
    """
    for attr in ("_assembly", "_similarity_metric", "_ceiling_func", "_ceiling"):
        if hasattr(child, attr):
            try:
                setattr(child, attr, None)
            except AttributeError:
                pass


def _child_raw_scalar(child_score) -> float:
    """Extract a scalar ``raw`` correlation from a child Score.

    brainscore_core's submission DB calls ``_retrieve_score_center(score.raw)``
    which assumes ``raw`` is reducible to a Python scalar. Children typically
    expose their uncieled raw correlation under ``attrs['raw']`` as a 0-d or
    1-d array; this helper coerces it to a float.
    """
    raw = child_score.attrs.get("raw")
    if raw is None:
        return float("nan")
    try:
        vals = raw.values if hasattr(raw, "values") else np.asarray(raw)
        return float(np.asarray(vals).mean())
    except Exception:
        return float("nan")

WRAPPER_N_BOOTSTRAP = 200


def _scaled_preallocate_memory(wrapper, candidate, child_factory, raise_if_oom: bool):
    """Probe one representative child and scale by ``wrapper._peak_aggregation``.

    Returns the child's unscaled :class:`MemoryEstimate` so callers can read
    formula details; raises :exc:`MemoryError` when the *scaled* estimate
    exceeds available RAM. Returns ``None`` when the probe itself is skipped
    (e.g. ``BRAINSCORE_SKIP_MEMORY_CHECK=1``).
    """
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory as _probe
    child = child_factory()
    try:
        estimate = _probe(candidate, child, raise_if_oom=False)
    finally:
        _release(child)
        del child
        gc.collect()
    if estimate is None:
        return None
    scaled_gb = estimate.total_estimated_gb * wrapper._peak_aggregation
    if scaled_gb > estimate.available_gb and raise_if_oom:
        raise MemoryError(
            f"preallocate_memory ({type(wrapper).__name__} '{wrapper.identifier}'): "
            f"aggregated estimate {scaled_gb:.1f} GB (per-child "
            f"{estimate.total_estimated_gb:.1f} × {wrapper._peak_aggregation}) "
            f"> {estimate.available_gb:.1f} GB available"
        )
    return estimate


def block_diagonal_concat(
    slices: list[xr.DataArray],
    presentation_coords: tuple[str, ...] = ("stimulus_id", "subject_id_pres", "repetition"),
    neuroid_coords: tuple[str, ...] = (
        "subject_id", "region", "roi", "voxel_index_in_brain_mask",
        "nc_4rep", "nc_12rep", "nc_allrep",
    ),
) -> xr.DataArray:
    """Concatenate per-subject slices block-diagonally on (presentation, neuroid).

    Each subject contributes its own presentation block on the diagonal; off-diagonal
    cells are NaN. Coords listed in ``presentation_coords`` and ``neuroid_coords``
    are concatenated 1D along their respective dims (i.e. they stay 1D, not 2D
    after the stack). Coords that are missing from any slice are silently dropped
    from the output so this works across heterogeneous slices.

    Why this exists: ``xr.concat(..., join='outer')`` on the neuroid dim broadcasts
    any 1D coord into a 2D representation when MultiIndex tuples differ across
    subjects, which blows out memory and corrupts coord semantics. Stitching by
    hand bypasses that.

    The default coord lists match the LAION-fMRI assembly shape; pass custom
    tuples for other datasets.
    """
    n_pres = [s.sizes["presentation"] for s in slices]
    n_neur = [s.sizes["neuroid"] for s in slices]
    total_pres, total_neur = sum(n_pres), sum(n_neur)

    dtype = slices[0].dtype
    data = np.full(
        (total_pres, total_neur),
        np.nan,
        dtype=dtype if dtype.kind == "f" else np.float32,
    )
    p_off, n_off = 0, 0
    for s, np_, nn in zip(slices, n_pres, n_neur):
        data[p_off : p_off + np_, n_off : n_off + nn] = s.values
        p_off += np_
        n_off += nn

    def _has(c, s):
        # MultiIndex levels are accessible via s[c] even though they're not in s.coords.
        try:
            s[c]
            return True
        except (KeyError, AttributeError):
            return False

    coords: dict = {}
    for c in presentation_coords:
        if all(_has(c, s) for s in slices):
            coords[c] = ("presentation", np.concatenate([s[c].values for s in slices]))
    for c in neuroid_coords:
        if all(_has(c, s) for s in slices):
            coords[c] = ("neuroid", np.concatenate([s[c].values for s in slices]))

    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

    da = xr.DataArray(data, dims=("presentation", "neuroid"), coords=coords, name="data")
    return NeuroidAssembly(da)


class KFoldNeuralBenchmark:
    """Aggregate fold benchmarks built on demand from a per-fold factory.

    Each fold's benchmark is built fresh, scored, then released before the next
    fold starts — peak memory stays at one fold's working set instead of N. The
    factory takes a 0-based fold index and returns a ready-to-call benchmark.

    Mirrors the public interface Brain-Score's ``load_benchmark`` + ``benchmark(model)``
    expects: ``identifier``, ``region``, ``parent``, ``bibtex``, ``ceiling``, and
    ``__call__(candidate) -> Score``.
    """

    def __init__(
        self,
        identifier: str,
        n_folds: int,
        fold_factory: Callable[[int], TrainTestNeuralBenchmark],
        peak_aggregation: float = 1.0,
    ):
        if n_folds < 1:
            raise ValueError("KFoldNeuralBenchmark requires at least one fold.")
        self.identifier = identifier
        self._n_folds = int(n_folds)
        self._factory = fold_factory
        # Folds run sequentially; default peak ≈ one fold. Override when the
        # activation cache or any cross-fold state accumulates in memory.
        self._peak_aggregation = float(peak_aggregation)

        # Materialize one fold to extract region/bibtex/version metadata, then release.
        sample = self._factory(0)
        self.region = sample.region
        self.parent = self.region
        self.bibtex = getattr(sample, "bibtex", None)
        self.version = getattr(sample, "version", 1)
        _release(sample)
        del sample
        gc.collect()

    def preallocate_memory(self, candidate, raise_if_oom: bool = True):
        return _scaled_preallocate_memory(
            wrapper=self, candidate=candidate,
            child_factory=lambda: self._factory(0),
            raise_if_oom=raise_if_oom,
        )

    @property
    def ceiling(self):
        values = np.empty(self._n_folds, dtype=np.float64)
        for k in range(self._n_folds):
            child = self._factory(k)
            values[k] = float(child.ceiling.values)
            _release(child)
            del child
            gc.collect()
        score = Score(values.mean())
        score.attrs["raw_folds"] = xr.DataArray(
            values, dims=("fold",), coords={"fold": np.arange(len(values))},
        )
        return score

    def __call__(self, candidate) -> Score:
        fold_scores = []
        fold_ceilings = np.empty(self._n_folds, dtype=np.float64)
        fold_raws = np.empty(self._n_folds, dtype=np.float64)
        for k in range(self._n_folds):
            child = self._factory(k)
            child_score = child(candidate)
            fold_scores.append(child_score)
            fold_ceilings[k] = float(child.ceiling.values)
            fold_raws[k] = _child_raw_scalar(child_score)
            _release(child)
            del child
            gc.collect()
        values = np.array([float(s.values) for s in fold_scores])
        score = Score(values.mean())
        raw_folds = xr.DataArray(
            values, dims=("fold",), coords={"fold": np.arange(len(values))},
            name=f"{self.identifier}_per_fold",
        )
        # Pre-set 'raw' as a scalar so attach_error won't overwrite it with the
        # disaggregated array; brainscore_core's DB recorder requires scalar.
        score.attrs["raw"] = float(np.nanmean(fold_raws))
        score.attrs["raw_folds"] = raw_folds
        score.attrs["sem_folds"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        ceiling = Score(fold_ceilings.mean())
        ceiling.attrs["raw_folds"] = xr.DataArray(
            fold_ceilings, dims=("fold",), coords={"fold": np.arange(len(fold_ceilings))},
        )
        score.attrs["ceiling"] = ceiling
        score.attrs["folds"] = fold_scores
        if len(values) > 1:
            score = attach_error(score, raw_folds, over=["fold"], n_bootstrap=WRAPPER_N_BOOTSTRAP)
        else:
            score = declare_no_error(score, reason="single fold; no resampleable axis at this layer")
        return score


class MultiSubjectNeuralBenchmark:
    """Run a per-subject ``TrainTestNeuralBenchmark``, aggregate across subjects.

    Brain-Score's built-in ``TrainTestNeuralBenchmark(alpha_coord='subject')``
    slices only the neuroid dim — that's fine when the matrix is dense (Allen2022:
    every presentation row contains every subject's voxels). For block-diagonal
    cross-subject designs (each row has signal only in one subject's voxels)
    per-subject training and scoring need both presentation and neuroid sliced.
    This wrapper instantiates one child benchmark per subject on a dense
    single-subject slice, runs them, and aggregates with mean + per-subject
    detail in ``score.attrs``.

    Mirrors the public interface Brain-Score's ``load_benchmark`` +
    ``benchmark(model)`` expects: ``identifier``, ``region``, ``parent``,
    ``bibtex``, ``timebins``, ``ceiling``, and ``__call__(candidate) -> Score``.
    """

    def __init__(
        self,
        identifier: str,
        subjects: Sequence[str],
        per_subject_factory: Callable[[str], TrainTestNeuralBenchmark],
        peak_aggregation: float = 1.0,
    ):
        if not subjects:
            raise ValueError("MultiSubjectNeuralBenchmark requires at least one subject.")
        self.identifier = identifier
        self._subjects = list(subjects)
        self._factory = per_subject_factory
        # Per-subject children run sequentially. Default peak ≈ one child;
        # raise when subjects have disjoint activation caches that accumulate
        # (e.g. LAION persubject pool: ~N× per-subject peak).
        self._peak_aggregation = float(peak_aggregation)

        # Build one child to extract metadata, then release. Subsequent children
        # are instantiated on demand inside ceiling/__call__ and dropped after use.
        sample = self._factory(self._subjects[0])
        self.region = sample.region
        self.parent = self.region
        self.bibtex = getattr(sample, "bibtex", None)
        self.version = getattr(sample, "version", 1)
        self.timebins = sample.timebins
        _release(sample)
        del sample
        gc.collect()

    def preallocate_memory(self, candidate, raise_if_oom: bool = True):
        return _scaled_preallocate_memory(
            wrapper=self, candidate=candidate,
            child_factory=lambda: self._factory(self._subjects[0]),
            raise_if_oom=raise_if_oom,
        )

    @property
    def ceiling(self):
        values = np.empty(len(self._subjects), dtype=np.float64)
        for i, sub_id in enumerate(self._subjects):
            child = self._factory(sub_id)
            values[i] = float(child.ceiling.values)
            _release(child)
            del child
            gc.collect()
        score = Score(values.mean())
        score.attrs["raw_subjects"] = xr.DataArray(
            values, dims=("subject",), coords={"subject": self._subjects},
        )
        score.attrs["sem"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return score

    def __call__(self, candidate) -> Score:
        per_subject_scores = []
        ceil_values = np.empty(len(self._subjects), dtype=np.float64)
        raw_values = np.empty(len(self._subjects), dtype=np.float64)
        for i, sub_id in enumerate(self._subjects):
            child = self._factory(sub_id)
            child_score = child(candidate)
            per_subject_scores.append(child_score)
            ceil_values[i] = float(child.ceiling.values)
            raw_values[i] = _child_raw_scalar(child_score)
            _release(child)
            del child
            gc.collect()
        values = np.array([float(s.values) for s in per_subject_scores])
        score = Score(values.mean())
        raw_subjects = xr.DataArray(
            values, dims=("subject",), coords={"subject": self._subjects},
            name=f"{self.identifier}_per_subject",
        )
        # Pre-set 'raw' as a scalar so attach_error won't overwrite it with the
        # disaggregated array; brainscore_core's DB recorder requires scalar.
        score.attrs["raw"] = float(np.nanmean(raw_values))
        score.attrs["raw_subjects"] = raw_subjects
        score.attrs["sem_subjects"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        ceiling = Score(ceil_values.mean())
        ceiling.attrs["raw_subjects"] = xr.DataArray(
            ceil_values, dims=("subject",), coords={"subject": self._subjects},
        )
        ceiling.attrs["sem"] = (
            float(ceil_values.std(ddof=1) / np.sqrt(len(ceil_values))) if len(ceil_values) > 1 else 0.0
        )
        score.attrs["ceiling"] = ceiling
        for sub_id, s in zip(self._subjects, per_subject_scores):
            score.attrs[sub_id] = s
        if len(values) > 1:
            score = attach_error(score, raw_subjects, over=["subject"], n_bootstrap=WRAPPER_N_BOOTSTRAP)
        else:
            score = declare_no_error(score, reason="single-subject wrapper; nothing to resample at this layer")
        return score
