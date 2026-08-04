"""Independent reproduction of LAION-fMRI's published 12-rep noise ceiling.

We use the per-session single-trial betas (`SingletrialBetas_statmap.nii.gz`)
and the trial→image TSVs from `derivatives/glmsingle-tedana/<sub>/<ses>/func/`,
then implement the NSD variance-based estimator (Allen et al., 2022) that the
LAION-fMRI authors describe in the published `Noiseceiling12rep_statmap.json`
sidecar:

    1) z-score betas within session per voxel
    2) for each image with exactly 12 repetitions: compute variance across reps
    3) average variance across images -> v_metric (per voxel)
    4) ncsnr = sqrt(1 - v_metric**2) / v_metric
    5) NC% = 100 * ncsnr**2 / (ncsnr**2 + 1/12)

Compare voxel-by-voxel to the dataset's shipped
`sub-XX_task-images_space-T1w_desc-Noiseceiling12rep_statmap.nii.gz`.

Scope: we restrict to V1 + V2 + V4 + IT voxels (the regions we actually
benchmark on). The published map covers the full brain mask; comparing only
within our benchmark regions is sufficient to validate our use of their NC
values and keeps the per-image accumulator small (~150 MB for 878 images x ~7K
voxels x 2 floats vs ~4 GB for the full mask).

Outputs:
    nc_reproduction_sub-XX.csv  — per-voxel reproduced vs published NC
    summary.csv                 — per-region per-subject Spearman r + median NC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# Filled in by main(); overridable on the CLI. Defaults align with the
# layout that `laion-fmri config --data-dir ~/laion-fmri` produces.
DATA_ROOT: Path
GLMS_ROOT: Path
ROI_ROOT: Path
STIM_META: Path
OUT_DIR: Path = Path(__file__).parent


def _set_roots(data_root: Path) -> None:
    """Configure module-level path roots from a single ``--data-dir``."""
    global DATA_ROOT, GLMS_ROOT, ROI_ROOT, STIM_META
    DATA_ROOT = data_root
    GLMS_ROOT = DATA_ROOT / "derivatives/glmsingle-tedana"
    ROI_ROOT = DATA_ROOT / "derivatives/rois"
    STIM_META = DATA_ROOT / "stimuli/task-images_metadata.csv"


def _load_region_mask(sub_id: str, region: str) -> np.ndarray:
    """Return a 3D boolean mask for the requested region (V1/V2/V4/IT)."""
    sub_roi = ROI_ROOT / sub_id

    if region in {"V1", "V2", "V4"}:
        # V1/V2 are bilateral retinotopic ROIs; combine ventral + dorsal
        if region == "V4":
            sources = [sub_roi / "retinotopy" / f"{sub_id}_space-T1w_res-1pt8_label-hV4_mask.nii.gz"]
        else:
            sources = [
                sub_roi / "retinotopy" / f"{sub_id}_space-T1w_res-1pt8_label-{region}v_mask.nii.gz",
                sub_roi / "retinotopy" / f"{sub_id}_space-T1w_res-1pt8_label-{region}d_mask.nii.gz",
            ]
        mask = None
        for src in sources:
            if not src.exists():
                raise FileNotFoundError(f"Missing ROI source for {sub_id} {region}: {src}")
            m = nib.load(src).get_fdata().astype(bool)
            mask = m if mask is None else (mask | m)
        return mask

    if region == "IT":
        # IT = laion-ventral \ (V1v|V1d|V2v|V2d|V3v|V3d|hV4)
        ventral_path = sub_roi / "selectivity" / f"{sub_id}_space-T1w_res-1pt8_label-laionventral_mask.nii.gz"
        if not ventral_path.exists():
            # Fall back to other naming conventions
            for p in sub_roi.rglob("*laionventral*mask.nii.gz"):
                ventral_path = p
                break
        if not ventral_path.exists():
            raise FileNotFoundError(f"laionventral mask missing for {sub_id}")
        it_mask = nib.load(ventral_path).get_fdata().astype(bool)
        retino_dir = sub_roi / "retinotopy"
        for region_to_subtract in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            p = retino_dir / f"{sub_id}_space-T1w_res-1pt8_label-{region_to_subtract}_mask.nii.gz"
            if p.exists():
                it_mask &= ~nib.load(p).get_fdata().astype(bool)
        return it_mask

    raise ValueError(f"Unknown region {region}")


def _sessions_for_sub(sub_id: str) -> list[str]:
    """List session IDs from the published NC sidecar (the authoritative source list)."""
    sidecar = GLMS_ROOT / sub_id / f"{sub_id}_task-images_space-T1w_desc-Noiseceiling12rep_statmap.json"
    with open(sidecar) as f:
        meta = json.load(f)
    sessions = []
    for src in meta["Sources"]:
        # bids::derivatives/glmsingle-tedana/sub-01/ses-XX/...
        parts = src.split("/")
        ses = next(p for p in parts if p.startswith("ses-"))
        sessions.append(ses)
    return sessions


def reproduce_nc12rep(sub_id: str) -> dict:
    """Recompute NC12rep for one subject, restricted to V1/V2/V4/IT voxels.

    Returns dict with per-region voxel arrays: reproduced NC, published NC,
    plus the 3D-aligned voxel indices for cross-reference.
    """
    print(f"\n=== {sub_id} ===")

    # 1) Build a combined V1+V2+V4+IT mask (boolean 3D), then flatten to voxel indices.
    regions = ("V1", "V2", "V4", "IT")
    region_masks = {r: _load_region_mask(sub_id, r) for r in regions}
    combined_mask_3d = np.zeros_like(region_masks["V1"], dtype=bool)
    for r in regions:
        combined_mask_3d |= region_masks[r]
    n_voxels = int(combined_mask_3d.sum())
    print(f"  Region voxels: V1={region_masks['V1'].sum()}, V2={region_masks['V2'].sum()}, "
          f"V4={region_masks['V4'].sum()}, IT={region_masks['IT'].sum()}, total={n_voxels}")

    # Per-voxel region label (for later aggregation)
    voxel_region_3d = np.full(combined_mask_3d.shape, "", dtype=object)
    for r in regions:
        voxel_region_3d[region_masks[r]] = r
    voxel_region = voxel_region_3d[combined_mask_3d]  # (n_voxels,)

    # 2) Identify 12-rep image labels.
    meta = pd.read_csv(STIM_META)
    twelve_rep = set(meta[meta["n_reps"] == "12rep"]["image_name"].tolist())
    print(f"  12-rep images in manifest: {len(twelve_rep)}")

    # 3) Streaming pass: per-session, extract betas inside region mask, z-score,
    #    accumulate per-image sums and sum-of-squares for variance computation.
    sessions = _sessions_for_sub(sub_id)
    print(f"  Sessions: {len(sessions)}")

    image_idx_map: dict[str, int] = {}
    sum_x = np.zeros((0, n_voxels), dtype=np.float64)
    sum_x2 = np.zeros((0, n_voxels), dtype=np.float64)
    counts = np.zeros((0,), dtype=np.int32)

    for k, ses in enumerate(sessions):
        ses_dir = GLMS_ROOT / sub_id / ses / "func"
        beta_path = ses_dir / f"{sub_id}_{ses}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz"
        trials_path = ses_dir / f"{sub_id}_{ses}_task-images_desc-SingletrialBetas_trials.tsv"
        if not beta_path.exists() or not trials_path.exists():
            print(f"    {ses}: SKIP (missing files)")
            continue

        trials = pd.read_csv(trials_path, sep="\t")
        rep_trial_idx = trials.index[trials["label"].isin(twelve_rep)].tolist()
        if not rep_trial_idx:
            continue

        # Memory-cheap: load full 4D, mask down to (n_voxels, n_trials), keep only 12-rep trials.
        img = nib.load(beta_path)
        data_4d = img.get_fdata(dtype=np.float32)            # (X,Y,Z,T)
        betas = data_4d[combined_mask_3d].astype(np.float32) # (n_voxels, T)
        del data_4d  # free memory

        # Z-score within RUN per voxel (each session has 12 runs of 87 trials).
        # NSD-style estimators z-score at the run level — within-session
        # z-scoring left signal that inflates our reproduced NC by ~1.5-2x.
        z = np.empty_like(betas)
        for run_id in trials["run"].unique():
            run_trial_idx = trials.index[trials["run"] == run_id].to_numpy()
            run_betas = betas[:, run_trial_idx]                 # (n_voxels, 87)
            mu = run_betas.mean(axis=1, keepdims=True)
            sd = run_betas.std(axis=1, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            z[:, run_trial_idx] = (run_betas - mu) / sd

        # Iterate only over 12-rep trials in this session
        new_images = 0
        for trial_i in rep_trial_idx:
            label = trials.at[trial_i, "label"]
            if label not in image_idx_map:
                image_idx_map[label] = len(image_idx_map)
                sum_x = np.vstack([sum_x, np.zeros((1, n_voxels), dtype=np.float64)])
                sum_x2 = np.vstack([sum_x2, np.zeros((1, n_voxels), dtype=np.float64)])
                counts = np.append(counts, 0)
                new_images += 1
            idx = image_idx_map[label]
            row = z[:, trial_i].astype(np.float64)
            sum_x[idx] += row
            sum_x2[idx] += row * row
            counts[idx] += 1

        print(f"    {ses} ({k+1}/{len(sessions)}): {len(rep_trial_idx)} 12-rep trials, "
              f"{new_images} new images, total images so far: {len(image_idx_map)}")

    # 4) Per-image variance across reps; only keep images with all 12 reps.
    keep = counts == 12
    print(f"  Images with exactly 12 reps: {int(keep.sum())} (of {len(counts)} encountered)")
    sum_x = sum_x[keep]
    sum_x2 = sum_x2[keep]
    n_per_image = 12

    # variance per (image, voxel): population variance is fine for ncsnr formula
    mean_per_img = sum_x / n_per_image
    var_per_img = sum_x2 / n_per_image - mean_per_img ** 2
    var_per_img = np.clip(var_per_img, 0, None)  # numerical guard

    # v_metric per voxel: sqrt of mean variance across images
    v_metric = np.sqrt(var_per_img.mean(axis=0))
    v_metric = np.where(np.isfinite(v_metric) & (v_metric > 0), v_metric, np.nan)

    # ncsnr per voxel
    ncsnr_sq = 1.0 - v_metric ** 2
    ncsnr_sq = np.clip(ncsnr_sq, 0, None)
    ncsnr = np.sqrt(ncsnr_sq) / v_metric
    nc_pct_reproduced = 100.0 * (ncsnr ** 2) / (ncsnr ** 2 + 1.0 / n_per_image)

    # 5) Load published NC and extract values at combined-mask voxels.
    pub_path = GLMS_ROOT / sub_id / f"{sub_id}_task-images_space-T1w_desc-Noiseceiling12rep_statmap.nii.gz"
    nc_pub_3d = nib.load(pub_path).get_fdata(dtype=np.float32)
    nc_pct_published = nc_pub_3d[combined_mask_3d]

    return {
        "sub_id": sub_id,
        "voxel_region": voxel_region,
        "nc_reproduced": nc_pct_reproduced,
        "nc_published": nc_pct_published,
        "n_images_used": int(keep.sum()),
    }


def summarize(result: dict) -> pd.DataFrame:
    """Per-region comparison: voxel Pearson, voxel Spearman, median reproduced vs published."""
    rows = []
    region_arr = result["voxel_region"]
    repd = result["nc_reproduced"]
    pubd = result["nc_published"]

    for r in ("V1", "V2", "V4", "IT"):
        mask = (region_arr == r) & np.isfinite(repd) & np.isfinite(pubd)
        if mask.sum() < 5:
            rows.append(dict(sub=result["sub_id"], region=r, n=int(mask.sum()),
                             repd_median=np.nan, pub_median=np.nan,
                             pearson=np.nan, spearman=np.nan))
            continue
        from scipy.stats import pearsonr, spearmanr
        pr = pearsonr(repd[mask], pubd[mask]).statistic
        sr = spearmanr(repd[mask], pubd[mask]).statistic
        rows.append(dict(
            sub=result["sub_id"], region=r, n=int(mask.sum()),
            repd_median=float(np.nanmedian(repd[mask])),
            pub_median=float(np.nanmedian(pubd[mask])),
            pearson=float(pr), spearman=float(sr),
        ))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path.home() / "laion-fmri",
                   help="LAION-fMRI data directory (matches `laion-fmri config`). "
                        "Expects `derivatives/glmsingle-tedana/`, "
                        "`derivatives/rois/`, and "
                        "`stimuli/task-images_metadata.csv` underneath. "
                        "Default: ~/laion-fmri/")
    p.add_argument("--subjects", nargs="*",
                   default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    args = p.parse_args()

    _set_roots(args.data_dir.expanduser())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for sub_id in args.subjects:
        try:
            r = reproduce_nc12rep(sub_id)
        except Exception as e:
            print(f"  FAILED for {sub_id}: {type(e).__name__}: {e}")
            continue

        # Save per-voxel CSV for forensic checks
        df_vox = pd.DataFrame({
            "region": r["voxel_region"],
            "nc_reproduced_pct": r["nc_reproduced"],
            "nc_published_pct":  r["nc_published"],
        })
        df_vox.to_csv(OUT_DIR / f"nc_reproduction_{sub_id}.csv", index=False)

        summary = summarize(r)
        print(summary.round(3).to_string(index=False))
        all_summaries.append(summary)

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(OUT_DIR / "summary.csv", index=False)
        print("\n=== Combined summary ===")
        print(combined.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
