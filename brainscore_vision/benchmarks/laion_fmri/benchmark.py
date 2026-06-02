"""LAION-fMRI Brain-Score benchmark.

Wraps the LAION-fMRI 7T volumetric dataset (Zerbe et al., VSS 2026) using the
bundled re:vision train/test splits (`tau`, `ood`, `cluster_k5_*`, plus per-OOD-category
sub-variants) on the Allen2022-style **shared pool** (1,492 stimuli seen by every
subject). Mirrors `Hebart2023_fmri` and `Allen2022_fmri` in benchmark structure.

Data design: ONE assembly per subject is registered in the data registry (e.g.
`LAION_fMRI_full_sub-01`); the benchmark loader iterates subjects, applies split +
region + NC filters per subject, then concatenates the small filtered slices on the
presentation dim only. Avoids materializing a 5+ GB NaN-padded cross-subject matrix
in RAM (which OOMed during development).

For cluster CV the `KFoldNeuralBenchmark` wrapper runs each of the 5 cluster folds as a
child `TrainTestNeuralBenchmark` and aggregates with mean + per-fold detail in attrs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from brainscore_core.metrics import Score
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision import load_dataset, load_metric, load_stimulus_set
from brainscore_vision.benchmark_helpers.multi_subject import (
    KFoldNeuralBenchmark,
    MultiSubjectNeuralBenchmark,
    WRAPPER_N_BOOTSTRAP,
    block_diagonal_concat,
)
from brainscore_vision.benchmark_helpers.neural_common import (
    TrainTestNeuralBenchmark,
    RSABenchmark,
    average_repetition,
    filter_reliable_neuroids,
    place_on_screen,
    timebins_from_assembly,
)
from brainscore_vision.metric_helpers.bootstrap_error import (
    attach_error,
    declare_no_error,
)
from brainscore_vision.utils import LazyLoad


# 21 log-spaced alphas covering 1e-10 to 1e10. Wider range than the brain-score
# default ALPHA_LIST so RidgeCV catches extreme regularization needs for very
# wide-feature models without disrupting other benchmarks' alpha conventions.
LAION_ALPHA_LIST = [10.0 ** i for i in range(-10, 11)]


BIBTEX = """@inproceedings{zerbe_laion-fmri_2026,
    title = {{LAION}-{fMRI}: A densely sampled 7T-fMRI dataset providing broad coverage of natural image diversity},
    author = {Zerbe, Josefine and Roth, Johannes and Mell, Maggie Mae and Herholz, Peer and Knapen, Tomas and Hebart, Martin N.},
    year = {2026},
    booktitle = {Vision Sciences Society Annual Meeting},
}"""

# Stimuli rendered 1000x1000 px on a BenQ-mirrored PROpixx projection at ~165 cm viewing distance
# -> 9.2 x 9.2 degrees of visual angle (LAION-fMRI experimental design docs).
VISUAL_DEGREES = 9.2

# Voxels with nc_12rep below this value are dropped before scoring.
# 30% retains majority of voxels per region for sub-01 (validated 2026-05-18).
NOISE_CEILING_THRESHOLD = 0.3 * 100

# Pool used by laion_fmri.splits when computing train/test masks.
# "all-subjects" pool: concatenate per-subject pools. Each subject's split set is the
# union of their per-subject pool plus the shared pool, so we mask against `sub-XX`
# for that subject's trials.
REGIONS = ("V1", "V2", "V3", "V4", "IT", "IT_full")
# `IT_full` runtime alias = `V4 ∪ IT` — approximates raw `laion-ventral` (the dataset
# authors' broader ventral mask) since 97.8% of hV4 voxels live inside laion-ventral.
# Our default `IT` uses `laion-ventral \ retinotopic` (matches NSD/Algonauts
# streams_ventral convention). Per the LAION-fMRI manual, "mutual exclusivity within
# set, overlap permitted across sets" — both definitions are valid; we expose both.
_RUNTIME_REGION_ALIAS = {
    "IT_full": ("V4", "IT"),
}
# When a benchmark's `region` doesn't appear in a model's region_layer_map, we
# substitute this canonical name for the model-side lookup. The benchmark identifier
# still carries the original (e.g. `IT_full`), but `candidate.start_recording(...)` uses
# the canonical region so existing model layer-maps keep working.
_MODEL_REGION_CANONICAL = {
    "IT_full": "IT",
}

OOD_TYPES = (
    "shape",
    "relations",
    "unusual",
    "cropped",
    "selfmade",
    "gaudy",
    "illusion-classic",
    "gabor",
    "illusion-natural",
)

SIMPLE_SPLITS = ("tau", "ood") + tuple(f"ood_{t}" for t in OOD_TYPES)
KFOLD_SPLITS = ("cluster_k5",)
ALL_SPLITS = SIMPLE_SPLITS + KFOLD_SPLITS

# Subjects included in the cross-subject shared-pool benchmark. Mirrors the
# `SUBJECTS` tuple in `data/laion_fmri/__init__.py` — keep in sync.
DEFAULT_SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


# ──────────────────────────────────────────────────────────────────────────
# Assembly loader (per-subject load + concat-on-presentation)
# ──────────────────────────────────────────────────────────────────────────


def _resolve_split(split: str) -> tuple[str, tuple[str, ...] | None]:
    """Map a benchmark-side split name to (canonical_split, ood_types_filter).

    Examples:
      tau                  -> ("tau", None)
      ood                  -> ("ood", None)
      ood_shape            -> ("ood", ("shape",))
      cluster_k5_0         -> ("cluster_k5_0", None)
    """
    if split.startswith("ood_") and split != "ood":
        return "ood", (split[len("ood_") :],)
    return split, None


def _split_mask_for_subject(trials: pd.DataFrame, split: str,
                              pool: str = "shared") -> tuple[np.ndarray, np.ndarray]:
    """Compute train/test masks for one subject's trials using the named re:vision pool.

    `pool="shared"` uses the cross-subject 1,492-image pool (Allen2022-style).
    `pool="sub-XX"` uses that subject's full 5,833-image pool (1,121 shared + 4,712 unique).
    """
    from laion_fmri.splits import get_split_masks

    canonical_split, ood_filter = _resolve_split(split)
    kw = {"ood_types": list(ood_filter)} if ood_filter is not None else {}
    return get_split_masks(trials, canonical_split, pool=pool, **kw)


# _block_diagonal_concat, KFoldNeuralBenchmark, and MultiSubjectNeuralBenchmark
# were extracted to brainscore_vision.benchmark_helpers.multi_subject so other
# block-diagonal multi-subject benchmarks can reuse them. See that module for
# the LAION-fMRI-specific coord defaults this benchmark relies on.


def _load_one_subject(
    sub_id: str,
    region: str,
    split: str,
    side: str,
    dataset_prefix: str,
    noise_ceiling_threshold: float,
    noise_ceiling_coord: str,
) -> xr.DataArray:
    """Load one subject's shared-pool slice, filtered down to (region, split, side).

    Heavy data stays subject-local: load -> filter neuroids by NC -> select region ->
    apply split mask -> return small slice. Peak per-subject memory ~150 MB.
    """
    full = load_dataset(f"{dataset_prefix}_full_{sub_id}")
    full = filter_reliable_neuroids(full, noise_ceiling_threshold, noise_ceiling_coord)
    # Resolve runtime aliases (e.g. IT_full = V4 ∪ IT) into a union of stored regions.
    if region in _RUNTIME_REGION_ALIAS:
        keep_regions = _RUNTIME_REGION_ALIAS[region]
        mask = np.isin(full["region"].values, keep_regions)
        full = full.isel(neuroid=np.flatnonzero(mask))
    else:
        full = full.sel(region=region)
    full.load()
    # `sel(region=...)` collapses the `region` level out of the neuroid MultiIndex,
    # and `isel(neuroid=...)` keeps it with mixed values. To re-stamp `region` as a
    # uniform per-neuroid coord pointing at the *requested* benchmark region, we must
    # first drop the old `region` level from the MultiIndex (if present) before
    # assign_coords. Otherwise xarray refuses ("cannot drop or update coordinate that
    # would corrupt the index").
    if "region" in (full.indexes.get("neuroid").names if "neuroid" in full.indexes else ()):
        # Reset the index, drop region, rebuild without it
        kept_levels = [lv for lv in full.indexes["neuroid"].names if lv != "region"]
        full = full.reset_index("neuroid").drop_vars("region", errors="ignore")
        if kept_levels:
            full = full.set_index(neuroid=kept_levels)
    elif "region" in full.coords:
        full = full.drop_vars("region")
    full = full.assign_coords(region=("neuroid", np.full(full.sizes["neuroid"], region, dtype=object)))
    if "time_bin" in full.dims:
        full = full.isel(time_bin=0)

    trials = pd.DataFrame(
        {
            "label": full["stimulus_id"].values,
            "repetition": full["repetition"].values if "repetition" in full.coords else 0,
        }
    )
    # Pool depends on dataset_prefix:
    #   "LAION_fMRI"           → "shared" pool (1,492 stimuli)
    #   "LAION_fMRI_persubject" → that subject's pool (5,833 stimuli)
    pool = sub_id if "persubject" in dataset_prefix else "shared"
    train_mask, test_mask = _split_mask_for_subject(trials, split, pool=pool)
    mask = train_mask if side == "train" else test_mask
    if not mask.any():
        raise ValueError(
            f"Empty {side} slice for {sub_id}, region={region}, split={split}, pool={pool}."
        )
    return full.isel(presentation=np.flatnonzero(mask))


def load_assembly(
    region: str,
    split: str,
    side: str,
    average_repetitions: bool,
    dataset_prefix: str = "LAION_fMRI",
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
):
    """Load one slice (train or test) of the cross-subject shared-pool assembly.

    Loads each subject's per-subject .nc, applies region + split + NC filters in-place,
    then concatenates the small filtered slices on the presentation dim. Because each
    subject's neuroids are independent (different physical voxels), the concat uses an
    outer join on neuroid -> the resulting matrix is NaN off the per-subject block
    diagonal. `TrainTestNeuralBenchmark(alpha_coord='subject')` slices both dims by
    subject_id before fitting, recovering dense per-subject matrices.
    """
    per_subject_slices: list[xr.DataArray] = []
    for sub_id in subjects:
        sliced = _load_one_subject(
            sub_id=sub_id, region=region, split=split, side=side,
            dataset_prefix=dataset_prefix,
            noise_ceiling_threshold=noise_ceiling_threshold,
            noise_ceiling_coord=noise_ceiling_coord,
        )
        per_subject_slices.append(sliced)

    # Manual block-diagonal concat:
    #   xr.concat with `join='outer'` on the neuroid dim broadcasts our 1D `region`
    #   coord to 2D because the MultiIndex tuples differ across subjects. Stitch the
    #   combined matrix by hand instead — placing each subject's data on its own
    #   neuroid block and leaving off-diagonal positions as NaN.
    combined = block_diagonal_concat(per_subject_slices)

    # Brain-Score's TrainTestNeuralBenchmark requires a `time_bin` dim with start/end
    # coords; fMRI has a single timepoint so add a singleton.
    combined = combined.expand_dims(time_bin=1).assign_coords(
        time_bin_start=("time_bin", [0]),
        time_bin_end=("time_bin", [0]),
        neuroid_id=("neuroid", np.arange(combined.sizes["neuroid"], dtype=np.int64)),
    )
    # Make `assembly['time_bin'].values` return (start, end) tuples so the model's
    # temporal look_at can unpack them. Allen2022 / Hebart2023 do the same.
    combined = combined.set_index(time_bin=["time_bin_start", "time_bin_end"])
    # Drop the time_bin dim (single timepoint for fMRI) while keeping the coord
    # so timebins_from_assembly still returns [(0, 0)] and ridge gets a 2D matrix.
    combined = combined.isel(time_bin=0)

    # Attach the stimulus set, sliced to just the stimuli present on this side.
    full_stim = load_stimulus_set(f"{dataset_prefix}_stim_full")
    side_image_ids = set(combined["stimulus_id"].values.tolist())
    stim = StimulusSet(full_stim[full_stim["stimulus_id"].isin(side_image_ids)].reset_index(drop=True))
    # Include the subject set in the identifier so per-subject benchmarks don't
    # collide in the model's activations cache. Shared-pool stim_ids are identical
    # across subjects (same cache hit = correct), but persubject stim_ids are
    # subject-specific — without this, sub-04 would replay sub-01's cached activations.
    # subject_id_pres is a MultiIndex level on `presentation`, so `in coords` returns
    # False — fetch via try/except instead.
    try:
        subs_in_assembly = sorted(set(combined["subject_id_pres"].values.tolist()))
    except (KeyError, AttributeError):
        subs_in_assembly = []
    sub_tag = "+".join(subs_in_assembly) if subs_in_assembly else "all"
    stim.identifier = f"{dataset_prefix}_{split}_{side}_{sub_tag}"
    stim.stimulus_paths = {
        sid: full_stim.stimulus_paths[sid]
        for sid in stim["stimulus_id"]
        if sid in full_stim.stimulus_paths
    }
    combined.attrs["stimulus_set"] = stim
    combined.attrs["stimulus_set_identifier"] = stim.identifier

    if average_repetitions:
        combined = average_repetition(combined)
    return combined


# ──────────────────────────────────────────────────────────────────────────
# Single-split factory (tau, ood, ood_<type>, cluster_k5_<k>)
# ──────────────────────────────────────────────────────────────────────────


def _ncsnr_ceiling(assembly, nc_coord: str = "nc_12rep"):
    """Use the dataset's published ncsnr-based noise ceiling as the benchmark ceiling.

    LAION-fMRI ships per-voxel NC maps computed from the noise-ceiling-SNR (ncsnr)
    statistic over repeated-stimulus variance. We convert the % R² value back to a
    correlation (sqrt) and expose it as a Brain-Score Score, with per-neuroid raw
    values preserved so `TrainTestNeuralBenchmark(alpha_coord='subject')` can slice
    per subject.

    This sidesteps the cross-subject NaN-padding incompatibility with
    `internal_consistency`'s split-half Pearson approach, and uses the
    publication-grade ceiling estimator the LAION-fMRI authors themselves provide.
    """
    nc_pct = assembly[nc_coord].values.astype(np.float32)  # 0-100 (%R²)
    nc_r = np.sqrt(np.clip(nc_pct, 0, 100) / 100.0)  # correlation scale

    raw = xr.DataArray(
        nc_r,
        dims=("neuroid",),
        coords={
            "neuroid_id": ("neuroid", assembly["neuroid_id"].values),
            "subject": ("neuroid", assembly["subject_id"].values),
            "region": ("neuroid", assembly["region"].values),
        },
    )
    aggregate = float(np.nanmedian(nc_r))
    score = Score(aggregate)
    score.attrs["raw"] = raw
    score.attrs["nc_coord"] = nc_coord
    return score


def _LAIONfMRI(
    region: str,
    split: str,
    similarity_metric,
    identifier_metric_suffix: str,
    dataset_prefix: str = "LAION_fMRI",
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    alpha_coord: str | None = None,
    per_voxel_ceilings: bool = False,
    visual_degrees: float = VISUAL_DEGREES,
    ceiler=None,
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
):
    if ceiler is None:
        ceiler = load_metric("internal_consistency")
    number_of_trials = 1

    # Train: average reps per stimulus so each row matches one model-activation row.
    # LAION-fMRI shared images have 4-12 reps each; without averaging the ridge fit
    # gets duplicate rows per stim and Brain-Score's row-aligned X/y matching fails.
    train_assembly = LazyLoad(
        lambda region=region, split=split, dp=dataset_prefix, subs=subjects,
        nct=noise_ceiling_threshold, ncc=noise_ceiling_coord:
        load_assembly(region, split, side="train", average_repetitions=True,
                      dataset_prefix=dp, subjects=subs,
                      noise_ceiling_threshold=nct, noise_ceiling_coord=ncc)
    )
    test_assembly = LazyLoad(
        lambda region=region, split=split, dp=dataset_prefix, subs=subjects,
        nct=noise_ceiling_threshold, ncc=noise_ceiling_coord:
        load_assembly(region, split, side="test", average_repetitions=True,
                      dataset_prefix=dp, subjects=subs,
                      noise_ceiling_threshold=nct, noise_ceiling_coord=ncc)
    )
    test_assembly_repetition = LazyLoad(
        lambda region=region, split=split, dp=dataset_prefix, subs=subjects,
        nct=noise_ceiling_threshold, ncc=noise_ceiling_coord:
        load_assembly(region, split, side="test", average_repetitions=False,
                      dataset_prefix=dp, subjects=subs,
                      noise_ceiling_threshold=nct, noise_ceiling_coord=ncc)
    )

    bench = TrainTestNeuralBenchmark(
        identifier=f"{dataset_prefix}.{region}-{split}-{identifier_metric_suffix}",
        version=1,
        ceiling_func=lambda: _ncsnr_ceiling(test_assembly_repetition, nc_coord=noise_ceiling_coord),
        train_assembly=train_assembly,
        test_assembly=test_assembly,
        similarity_metric=similarity_metric,
        alpha_coord=alpha_coord,
        per_voxel_ceilings=per_voxel_ceilings,
        visual_degrees=visual_degrees,
        number_of_trials=number_of_trials,
        parent=region,
        bibtex=BIBTEX,
    )
    # Substitute the model-side region name if this benchmark uses a non-canonical
    # region (e.g. IT_full -> IT). The benchmark's own identifier and the assembly's
    # `region` coord stay as-is; only `candidate.start_recording(region)` sees the
    # canonical name. This avoids per-model region_layer_map edits.
    if region in _MODEL_REGION_CANONICAL:
        bench.region = _MODEL_REGION_CANONICAL[region]
    return bench


def LAIONfMRI(
    region: str,
    split: str,
    metric_type: str = "ridgecv",
    dataset_prefix: str = "LAION_fMRI",
    alphas: Sequence[float] = LAION_ALPHA_LIST,
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
):
    """Construct a LAION-fMRI single-split benchmark (tau / ood / ood_<type>).

    For one subject: returns a plain `TrainTestNeuralBenchmark` on a dense slice.
    For multiple subjects: wraps one child benchmark per subject in a
    `MultiSubjectNeuralBenchmark` aggregator — each child runs on dense
    single-subject data, scores are averaged across subjects with per-subject
    detail preserved in `score.attrs`.

    The per-subject design (vs Allen2022's cross-subject dense rows) preserves
    LAION-fMRI's per-subject unique-image breadth and avoids the block-diagonal
    NaN-padding that breaks Brain-Score's `alpha_coord` machinery.

    :param noise_ceiling_threshold: voxels with ``nc_12rep`` below this fraction
        of explainable variance are filtered out before scoring. Default 0.30.
    :param noise_ceiling_coord: which published noise-ceiling estimate to filter
        by. Defaults to ``nc_12rep`` (12-repetition estimate, highest SNR).
    """
    if split not in SIMPLE_SPLITS:
        raise ValueError(
            f"split={split!r} is not a single-split variant. "
            f"Use LAIONfMRIClusterCV for cluster_k5. Available: {SIMPLE_SPLITS}"
        )
    similarity_metric = load_metric(f"dual_{metric_type}_split", alphas=alphas)
    if len(subjects) == 1:
        bench = _LAIONfMRI(
            region=region, split=split, similarity_metric=similarity_metric,
            identifier_metric_suffix=metric_type, dataset_prefix=dataset_prefix,
            subjects=subjects, alpha_coord=None, per_voxel_ceilings=False,
            noise_ceiling_threshold=noise_ceiling_threshold,
            noise_ceiling_coord=noise_ceiling_coord,
        )
        return _SingleSubjectErrorShim(bench)

    per_subject_benchmarks = [
        _LAIONfMRI(
            region=region, split=split, similarity_metric=similarity_metric,
            identifier_metric_suffix=metric_type, dataset_prefix=dataset_prefix,
            subjects=(sub_id,), alpha_coord=None, per_voxel_ceilings=False,
            noise_ceiling_threshold=noise_ceiling_threshold,
            noise_ceiling_coord=noise_ceiling_coord,
        )
        for sub_id in subjects
    ]
    identifier = f"{dataset_prefix}.{region}-{split}-{metric_type}"
    return MultiSubjectNeuralBenchmark(
        identifier=identifier, subjects=subjects,
        per_subject_benchmarks=per_subject_benchmarks,
    )


# KFoldNeuralBenchmark and MultiSubjectNeuralBenchmark were extracted to
# brainscore_vision.benchmark_helpers.multi_subject.


class _SingleSubjectErrorShim:
    _REASON = "single-subject TrainTest; regression metric exposes no per-stimulus axis"

    def __init__(self, child):
        self._child = child

    def __getattr__(self, name):
        return getattr(self._child, name)

    def __call__(self, candidate) -> Score:
        return declare_no_error(self._child(candidate), reason=self._REASON)


def LAIONfMRIClusterCV(
    region: str,
    metric_type: str = "ridgecv",
    dataset_prefix: str = "LAION_fMRI",
    alphas: Sequence[float] = LAION_ALPHA_LIST,
    n_folds: int = 5,
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
) -> KFoldNeuralBenchmark:
    """5-fold CLIP-cluster cross-validated benchmark.

    Each fold uses the `cluster_k5_<k>` split family from the LAION-fMRI re:vision
    initiative. For multi-subject runs, each fold is itself a `MultiSubjectNeuralBenchmark`
    (per-subject scoring inside each fold, then mean across folds at the outer level).
    Single-subject runs use a plain `TrainTestNeuralBenchmark` per fold.
    Per-fold scores live in ``score.attrs["raw_folds"]`` after evaluation.

    :param noise_ceiling_threshold: voxels with ``nc_12rep`` below this fraction
        of explainable variance are filtered out before scoring. Default 0.30.
    :param noise_ceiling_coord: which published noise-ceiling estimate to filter
        by. Defaults to ``nc_12rep`` (12-repetition estimate, highest SNR).
    """
    similarity_metric = load_metric(f"dual_{metric_type}_split", alphas=alphas)

    def _make_fold(k: int):
        if len(subjects) == 1:
            return _LAIONfMRI(
                region=region, split=f"cluster_k5_{k}",
                similarity_metric=similarity_metric,
                identifier_metric_suffix=metric_type,
                dataset_prefix=dataset_prefix, subjects=subjects,
                alpha_coord=None, per_voxel_ceilings=False,
                noise_ceiling_threshold=noise_ceiling_threshold,
                noise_ceiling_coord=noise_ceiling_coord,
            )
        children = [
            _LAIONfMRI(
                region=region, split=f"cluster_k5_{k}",
                similarity_metric=similarity_metric,
                identifier_metric_suffix=metric_type,
                dataset_prefix=dataset_prefix, subjects=(s,),
                alpha_coord=None, per_voxel_ceilings=False,
                noise_ceiling_threshold=noise_ceiling_threshold,
                noise_ceiling_coord=noise_ceiling_coord,
            )
            for s in subjects
        ]
        return MultiSubjectNeuralBenchmark(
            identifier=f"{dataset_prefix}.{region}-cluster_k5_{k}-{metric_type}",
            subjects=subjects, per_subject_benchmarks=children,
        )

    fold_benchmarks = [_make_fold(k) for k in range(n_folds)]
    identifier = f"{dataset_prefix}.{region}-cluster_k{n_folds}-{metric_type}"
    return KFoldNeuralBenchmark(identifier=identifier, fold_benchmarks=fold_benchmarks)


# ──────────────────────────────────────────────────────────────────────────
# RSA (representational similarity analysis) benchmark family
# ──────────────────────────────────────────────────────────────────────────


def load_full_assembly_one_subject(
    sub_id: str,
    region: str,
    dataset_prefix: str = "LAION_fMRI",
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
) -> xr.DataArray:
    """Load tau train + tau test for one subject as a single dense assembly.

    RSA needs the full per-subject stimulus pool with no train/test partition.
    We piggy-back on the existing per-side loader, then concatenate the two
    sides on the presentation dim. The result is a dense (n_pres, n_voxels)
    matrix — block-diagonal NaN padding never enters because there's only
    one subject here.

    The model RDM will be computed on this assembly's stimulus_set; the neural
    RDM on the betas. Stim_set carries a subject-tagged identifier so the
    model's activations cache stays correctly namespaced per subject.
    """
    train = load_assembly(
        region=region, split="tau", side="train", average_repetitions=True,
        dataset_prefix=dataset_prefix, subjects=(sub_id,),
        noise_ceiling_threshold=noise_ceiling_threshold,
        noise_ceiling_coord=noise_ceiling_coord,
    )
    test = load_assembly(
        region=region, split="tau", side="test", average_repetitions=True,
        dataset_prefix=dataset_prefix, subjects=(sub_id,),
        noise_ceiling_threshold=noise_ceiling_threshold,
        noise_ceiling_coord=noise_ceiling_coord,
    )

    combined = xr.concat([train, test], dim="presentation")

    # Build the stim_set covering both train + test stim. We can't `xr.concat`
    # stimulus_set DataFrames so combine via pandas.
    train_stim = train.attrs["stimulus_set"]
    test_stim = test.attrs["stimulus_set"]
    merged = pd.concat([train_stim, test_stim], ignore_index=True).drop_duplicates(
        subset="stimulus_id"
    ).reset_index(drop=True)
    stim = StimulusSet(merged)
    stim.identifier = f"{dataset_prefix}_rdm_full_{sub_id}"
    stim.stimulus_paths = {
        **train_stim.stimulus_paths,
        **test_stim.stimulus_paths,
    }
    combined.attrs["stimulus_set"] = stim
    combined.attrs["stimulus_set_identifier"] = stim.identifier

    # RSABenchmark expects the neuroid dim to carry a `subject` coord. Add it
    # as a MultiIndex *level* on the neuroid dim (not a regular coord) — that's
    # the convention Allen2022/Hebart2023 follow, and it sidesteps the
    # brainscore_vision RDM-metric coord-handling bug entirely (MultiIndex
    # levels are invisible to `assembly.coords.items()` so the metric's filter
    # doesn't need to know about them). Also future-proofs against the stricter
    # index-alignment behavior introduced in xarray 2025+.
    n_neur = combined.sizes["neuroid"]
    if "neuroid" in combined.indexes:
        existing_levels = list(combined.indexes["neuroid"].names)
        combined = combined.reset_index("neuroid")
    else:
        existing_levels = []
    combined = combined.assign_coords(
        subject=("neuroid", np.full(n_neur, sub_id, dtype=object))
    )
    levels = [lv for lv in existing_levels if lv != "neuroid"] + ["subject"]
    combined = combined.set_index(neuroid=levels)
    return combined


def _ncsnr_rsa_ceiler(assembly) -> Score:
    """Per-subject ncsnr-based RDM-reliability ceiling.

    Mean across the subject's neuroids of `sqrt(nc_12rep/100)` (ncsnr in
    correlation space). A loose proxy for the upper bound on how well a model
    RDM can correlate with this subject's neural RDM, derived from the
    dataset's published per-voxel reliability rather than computed via Nili's
    cross-subject method (which requires the same stimulus set across subjects
    and is provided separately via brainscore_vision's `rsa_ceiling` metric).
    """
    nc_pct = assembly["nc_12rep"].values.astype(np.float32)
    nc_r = np.sqrt(np.clip(nc_pct, 0, 100) / 100.0)
    ceiling = Score(float(np.nanmean(nc_r)))
    ceiling.attrs["ncsnr_voxel_count"] = int(np.isfinite(nc_r).sum())
    return ceiling


class _MultiSubjectRSABenchmark:
    """Run a per-subject :class:`RSABenchmark`, aggregate ceiled scores across subjects.

    Mirrors :class:`MultiSubjectNeuralBenchmark`'s aggregation contract but for
    RSA. Each child :class:`RSABenchmark` receives a dense single-subject
    assembly where ``subject`` is a neuroid MultiIndex level — so the upstream
    RDM metric sees a clean coords iterator and no workaround is needed.

    Could be promoted to ``benchmark_helpers/multi_subject.py`` once a second
    benchmark needs it.
    """

    def __init__(
        self,
        identifier: str,
        subjects: Sequence[str],
        per_subject_benchmarks: Sequence[RSABenchmark],
        region: str,
        bibtex: str = BIBTEX,
    ):
        if len(per_subject_benchmarks) != len(subjects):
            raise ValueError(
                f"subjects={list(subjects)} length disagrees with "
                f"per_subject_benchmarks length {len(per_subject_benchmarks)}"
            )
        self.identifier = identifier
        self.region = _MODEL_REGION_CANONICAL.get(region, region)
        self.parent = region
        self.bibtex = bibtex
        self._subjects = list(subjects)
        self._per_subject = list(per_subject_benchmarks)
        self.timebins = getattr(self._per_subject[0], "timebins", [(0, 0)])

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
        raw_subjects = xr.DataArray(
            values, dims=("subject",), coords={"subject": self._subjects},
            name=f"{self.identifier}_per_subject",
        )
        score.attrs["raw_subjects"] = raw_subjects
        score.attrs["sem_subjects"] = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        score.attrs["ceiling"] = self.ceiling
        for sub_id, s in zip(self._subjects, per_subject_scores):
            score.attrs[sub_id] = s
        if len(values) > 1:
            score = attach_error(
                score, raw_subjects, over=["subject"], n_bootstrap=WRAPPER_N_BOOTSTRAP,
            )
        else:
            score = declare_no_error(
                score, reason="single-subject RSA wrapper; nothing to resample at this layer",
            )
        return score


def LAIONfMRIRSA(
    region: str,
    dataset_prefix: str = "LAION_fMRI",
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS,
    noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD,
    noise_ceiling_coord: str = "nc_12rep",
) -> _MultiSubjectRSABenchmark:
    """Construct an RSA benchmark for one region across all subjects.

    RSA is only registered for the shared pool (``LAION_fMRI``). The persubject
    pool has subject-specific stimuli, which breaks cross-subject RDM
    comparison.

    Returns a thin :class:`_MultiSubjectRSABenchmark` wrapping one
    :class:`brainscore_vision.benchmark_helpers.neural_common.RSABenchmark`
    per subject. Each child operates on a dense single-subject assembly with
    ``subject`` carried as a neuroid MultiIndex level so brainscore_vision's
    RDM metric handles coord iteration cleanly out of the box.
    """
    if "persubject" in dataset_prefix:
        raise ValueError(
            "LAIONfMRIRSA is only defined for the shared pool. Persubject "
            "stimuli differ across subjects → no Nili ceiling."
        )

    per_subject_benchmarks = []
    for sub_id in subjects:
        assembly = LazyLoad(
            lambda sid=sub_id, r=region, dp=dataset_prefix,
            nct=noise_ceiling_threshold, ncc=noise_ceiling_coord:
            load_full_assembly_one_subject(
                sub_id=sid, region=r, dataset_prefix=dp,
                noise_ceiling_threshold=nct, noise_ceiling_coord=ncc,
            )
        )
        model_region = _MODEL_REGION_CANONICAL.get(region, region)
        bench = RSABenchmark(
            identifier=f"{dataset_prefix}.{region}-rdm-pearson-{sub_id}",
            version=1,
            assembly=assembly,
            region=model_region,
            visual_degrees=VISUAL_DEGREES,
            number_of_trials=1,
            bibtex=BIBTEX,
            ceiler=_ncsnr_rsa_ceiler,
            parent=region,
        )
        per_subject_benchmarks.append(bench)

    return _MultiSubjectRSABenchmark(
        identifier=f"{dataset_prefix}.{region}-rdm-pearson",
        subjects=subjects,
        per_subject_benchmarks=per_subject_benchmarks,
        region=region,
    )
