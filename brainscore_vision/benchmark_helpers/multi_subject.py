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

from typing import Sequence

import numpy as np
import xarray as xr

from brainscore_core import Score
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark


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
    """Aggregate a sequence of :class:`TrainTestNeuralBenchmark` instances across folds.

    Each child benchmark runs its own train/test cycle on its fold; this wrapper
    averages the per-fold ceiled scores and surfaces per-fold detail in
    ``score.attrs["raw_folds"]`` (an xr.DataArray indexed by fold).

    Mirrors the public interface Brain-Score's ``load_benchmark`` + ``benchmark(model)``
    expects: ``identifier``, ``region``, ``parent``, ``bibtex``, ``ceiling``, and
    ``__call__(candidate) -> Score``.
    """

    def __init__(self, identifier: str, fold_benchmarks: Sequence[TrainTestNeuralBenchmark]):
        if not fold_benchmarks:
            raise ValueError("KFoldNeuralBenchmark requires at least one fold benchmark.")
        self.identifier = identifier
        self._folds = list(fold_benchmarks)

        regions = {b.region for b in self._folds}
        if len(regions) != 1:
            raise ValueError(f"KFold folds disagree on region: {regions}")
        self.region = regions.pop()
        self.parent = self.region
        self.bibtex = getattr(self._folds[0], "bibtex", None)

    @property
    def ceiling(self):
        per_fold = [b.ceiling for b in self._folds]
        values = np.array([float(c.values) for c in per_fold])
        score = Score(values.mean())
        score.attrs["raw_folds"] = xr.DataArray(
            values, dims=("fold",), coords={"fold": np.arange(len(values))},
        )
        return score

    def __call__(self, candidate) -> Score:
        fold_scores = [b(candidate) for b in self._folds]
        values = np.array([float(s.values) for s in fold_scores])
        score = Score(values.mean())
        score.attrs["raw_folds"] = xr.DataArray(
            values, dims=("fold",), coords={"fold": np.arange(len(values))},
            name=f"{self.identifier}_per_fold",
        )
        score.attrs["sem_folds"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        score.attrs["ceiling"] = self.ceiling
        score.attrs["folds"] = fold_scores
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
        per_subject_benchmarks: Sequence[TrainTestNeuralBenchmark],
    ):
        if len(per_subject_benchmarks) != len(subjects):
            raise ValueError(
                f"subjects={list(subjects)} length disagrees with "
                f"per_subject_benchmarks length {len(per_subject_benchmarks)}"
            )
        self.identifier = identifier
        self._subjects = list(subjects)
        self._per_subject = list(per_subject_benchmarks)

        regions = {b.region for b in self._per_subject}
        if len(regions) != 1:
            raise ValueError(f"Subjects disagree on region: {regions}")
        self.region = regions.pop()
        self.parent = self.region
        self.bibtex = getattr(self._per_subject[0], "bibtex", None)
        self.timebins = self._per_subject[0].timebins

    @property
    def ceiling(self):
        per_subject = [b.ceiling for b in self._per_subject]
        values = np.array([float(c.values) for c in per_subject])
        score = Score(values.mean())
        score.attrs["raw_subjects"] = xr.DataArray(
            values, dims=("subject",), coords={"subject": self._subjects},
        )
        score.attrs["sem"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return score

    def __call__(self, candidate) -> Score:
        per_subject_scores = [child(candidate) for child in self._per_subject]
        values = np.array([float(s.values) for s in per_subject_scores])
        score = Score(values.mean())
        score.attrs["raw_subjects"] = xr.DataArray(
            values, dims=("subject",), coords={"subject": self._subjects},
            name=f"{self.identifier}_per_subject",
        )
        score.attrs["sem_subjects"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        score.attrs["ceiling"] = self.ceiling
        for sub_id, s in zip(self._subjects, per_subject_scores):
            score.attrs[sub_id] = s
        return score
