"""Build a LAION-fMRI NeuroidAssembly from locally-downloaded GLMsingle betas.

This script:
  1. Loads single-trial betas for one subject across all sessions (volumetric T1w 1.8mm).
  2. Restricts to a combined ROI mask covering V1/V2/V3/V4/IT (IT = laionventral minus retinotopic).
  3. Z-scores betas within session per voxel (ridge-regression magnitudes are otherwise non-comparable).
  4. Annotates neuroid coords (region, roi, voxel_index, noise_ceiling_4rep/12rep/allrep, ncsnr-derived NC, subject).
  5. Annotates presentation coords (stimulus_id, session, run, beta_index, repetition).
  6. Returns an xarray NeuroidAssembly ready for split-based subsetting and Brain-Score packaging.

Run separately per subject; outputs serialized to .nc for later cross-subject concatenation and S3 upload.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from laion_fmri.subject import load_subject

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_assembly")


REGION_DEFS = {
    "V1": ["V1v", "V1d"],
    "V2": ["V2v", "V2d"],
    "V3": ["V3v", "V3d"],
    "V4": ["hV4"],
    # IT is defined below as laionventral \ all retinotopic ROIs above.
}
IT_BASE = "laionventral"
RETINOTOPIC_SUB_ROIS = [r for rois in REGION_DEFS.values() for r in rois]


def build_region_assignment(sub) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (combined_mask, region_per_voxel, roi_per_voxel) over the subject's brain-mask voxel space.

    region_per_voxel labels each voxel in the combined mask with one of {V1, V2, V3, V4, IT};
    roi_per_voxel preserves the source sub-ROI (e.g. V1v, V1d, hV4, laionventral).
    """
    queries = list(RETINOTOPIC_SUB_ROIS) + [IT_BASE]
    masks = sub.get_roi_masks(queries=queries)
    n_voxels_brain = sub.get_n_voxels()

    combined = np.zeros(n_voxels_brain, dtype=bool)
    region_per_voxel = np.full(n_voxels_brain, "", dtype=object)
    roi_per_voxel = np.full(n_voxels_brain, "", dtype=object)

    retinotopic_union = np.zeros(n_voxels_brain, dtype=bool)
    for region, sub_rois in REGION_DEFS.items():
        for sub_roi in sub_rois:
            m = masks[sub_roi] > 0
            combined |= m
            region_per_voxel[m] = region
            roi_per_voxel[m] = sub_roi
            retinotopic_union |= m

    it_mask = (masks[IT_BASE] > 0) & ~retinotopic_union
    combined |= it_mask
    region_per_voxel[it_mask] = "IT"
    roi_per_voxel[it_mask] = IT_BASE

    return combined, region_per_voxel, roi_per_voxel


def zscore_within_session(betas_per_session: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score each voxel within each session.

    Ridge-regression beta magnitudes vary per-voxel and per-session, so cross-session pooling
    requires normalizing first (Allen2022/NSD precedent).
    """
    out = {}
    for ses, betas in betas_per_session.items():
        # Per the LAION-fMRI manual: "A small fraction of voxels near the brain mask edge
        # or in dropout regions may contain NaN; check for them." Use NaN-safe mean/std and
        # replace residual NaN/Inf with 0 so downstream metrics aren't poisoned.
        mu = np.nanmean(betas, axis=0, keepdims=True)
        sd = np.nanstd(betas, axis=0, keepdims=True)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        z = (betas - mu) / sd
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out[ses] = z.astype(np.float32)
    return out


def load_subject_data(subject_id: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a single subject end-to-end.

    Returns:
        betas:               (n_trials_total, n_voxels) float32, z-scored within session
        trials:              DataFrame with session/run/beta_index/label per row
        combined_mask:       (n_voxels_brain,) bool — which brain-mask voxels are in our ROI set
        region_per_voxel:    (n_voxels_in_mask,) object — V1/V2/V3/V4/IT labels (in mask order)
        roi_per_voxel:       (n_voxels_in_mask,) object — V1v/V1d/.../laionventral labels
        nc:                  dict with 'nc_4rep', 'nc_12rep', 'nc_allrep' — each (n_voxels_in_mask,)
    """
    sub = load_subject(subject_id)
    sessions = sub.get_sessions()
    log.info("%s: %d sessions", subject_id, len(sessions))

    combined_mask, region_full, roi_full = build_region_assignment(sub)
    n_in_mask = combined_mask.sum()
    log.info("Combined ROI voxels: %d", n_in_mask)
    region_per_voxel = region_full[combined_mask]
    roi_per_voxel = roi_full[combined_mask]

    for region in ["V1", "V2", "V3", "V4", "IT"]:
        log.info("  %s: %d voxels", region, (region_per_voxel == region).sum())

    betas_per_session: dict[str, np.ndarray] = {}
    trials_per_session: list[pd.DataFrame] = []
    for i, ses in enumerate(sessions, start=1):
        t0 = time.time()
        betas = sub.get_betas(session=ses, mask=combined_mask)
        trials = sub.get_trial_info(session=ses)
        betas_per_session[ses] = betas
        trials_per_session.append(trials)
        log.info("  [%d/%d] %s loaded in %.1fs  (betas %s, trials %s)",
                 i, len(sessions), ses, time.time() - t0, betas.shape, trials.shape)

    log.info("Z-scoring within session...")
    betas_per_session = zscore_within_session(betas_per_session)

    betas = np.concatenate([betas_per_session[s] for s in sessions], axis=0).astype(np.float32)
    trials = pd.concat(trials_per_session, ignore_index=True)
    log.info("Final betas: %s  trials: %s", betas.shape, trials.shape)

    nc = {}
    for desc, key in [("Noiseceiling4rep", "nc_4rep"),
                      ("Noiseceiling12rep", "nc_12rep"),
                      ("NoiseceilingAllrep", "nc_allrep")]:
        nc_full = sub.get_noise_ceiling(desc=desc)
        nc[key] = nc_full[combined_mask].astype(np.float32)
        log.info("%s (in mask): median %.2f, mean %.2f", desc, np.nanmedian(nc[key]), np.nanmean(nc[key]))

    return betas, trials, combined_mask, region_per_voxel, roi_per_voxel, nc


def build_assembly(subject_id: str) -> xr.DataArray:
    """Build a Brain-Score-style NeuroidAssembly (xr.DataArray with presentation + neuroid dims)."""
    betas, trials, combined_mask, region_per_voxel, roi_per_voxel, nc = load_subject_data(subject_id)

    n_trials, n_voxels = betas.shape
    rep_idx = trials.groupby("label").cumcount().values

    assembly = xr.DataArray(
        betas,
        dims=("presentation", "neuroid"),
        coords={
            "presentation": np.arange(n_trials),
            "stimulus_id": ("presentation", trials["label"].values),
            "session": ("presentation", trials["session"].values),
            "run": ("presentation", trials["run"].values),
            "beta_index": ("presentation", trials["beta_index"].values),
            "repetition": ("presentation", rep_idx),
            "neuroid": np.arange(n_voxels),
            "subject_id": ("neuroid", np.full(n_voxels, subject_id, dtype=object)),
            "region": ("neuroid", region_per_voxel),
            "roi": ("neuroid", roi_per_voxel),
            "voxel_index_in_brain_mask": ("neuroid", np.flatnonzero(combined_mask).astype(np.int32)),
            "nc_4rep": ("neuroid", nc["nc_4rep"]),
            "nc_12rep": ("neuroid", nc["nc_12rep"]),
            "nc_allrep": ("neuroid", nc["nc_allrep"]),
        },
        name=f"LAION_fMRI_{subject_id}",
    )
    return assembly


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. sub-01")
    p.add_argument("--out", required=True, help="output .nc path")
    args = p.parse_args()

    assembly = build_assembly(args.subject)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    assembly.to_netcdf(out)
    log.info("Wrote %s  (size: %.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
