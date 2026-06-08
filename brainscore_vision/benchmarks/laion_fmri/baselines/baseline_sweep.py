"""LAION-fMRI baseline sweep across the 20 headline benchmark variants.

Default lineup is the 5-model baseline reported in METHODS.md (alexnet,
alexnet_random, resnet50_tutorial, convnext_tiny, resnext101_32x8d_wsl) +
one additional shipped-layermap model (convnext_base). Edit ``MODELS``
below to swap in your own.

Run via:
    python -m brainscore_vision.benchmarks.laion_fmri.baselines.baseline_sweep
    # OR for one-cell-per-process (lighter memory):
    bash brainscore_vision/benchmarks/laion_fmri/baselines/sweep_per_cell.sh

Outputs (written alongside this script in ``baselines/``):
    baseline_sweep_results.csv   long form, one row per (model, benchmark) cell
    baseline_sweep_pivot.csv     wide pivot: benchmark x model

Idempotent: brain-score's result_caching short-circuits already-completed
(model, benchmark) cells. Failures per cell are captured (not fatal) so the
sweep finishes even when a single config errors.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


REGIONS = ["V1", "V2", "V4", "IT"]
SPLITS = ["tau", "ood"]                      # cluster_k5 demoted to API-only
FAMILIES = ["LAION_fMRI", "LAION_fMRI_persubject"]
MODELS = [
    # smaller baselines
    "alexnet",
    "alexnet_random",
    "resnet50_tutorial",
    # stronger models (run on EC2). Only models with SHIPPED region_layer_map
    # JSONs work reliably — auto-commit hits an xarray API mismatch on EC2's
    # newer xarray, so resnext101_32x16d_wsl and vitb14_dinov2_imagenet1k were
    # swapped for shipped-map alternatives in the same model families.
    "convnext_tiny_imagenet1K_torchvisionV1",
    "resnext101_32x8d_wsl",
    "convnext_base_fb_in22k_ft_in1k",
]

# 20 headline benchmarks: 16 ridge + 4 shared RSA
BENCHMARKS = (
    [f"{fam}.{r}-{s}-ridge" for fam in FAMILIES for r in REGIONS for s in SPLITS]
    + [f"LAION_fMRI.{r}-rdm-pearson" for r in REGIONS]
)

# CSVs land next to this script so the published baseline lives with the code.
OUT_DIR = Path(__file__).parent
LONG_CSV = OUT_DIR / "baseline_sweep_results.csv"
PIVOT_CSV = OUT_DIR / "baseline_sweep_pivot.csv"


def _resume_state() -> set[tuple[str, str]]:
    """Return (model, benchmark) tuples already in the long CSV so we can skip them."""
    if not LONG_CSV.exists():
        return set()
    df = pd.read_csv(LONG_CSV)
    return set(zip(df["model"], df["benchmark"]))


def _append_row(row: dict) -> None:
    write_header = not LONG_CSV.exists()
    pd.DataFrame([row]).to_csv(LONG_CSV, mode="a", header=write_header, index=False)


def run_one(model_id: str, bench_id: str, idx: int = 0, total: int = 0) -> None:
    """Run a single (model, benchmark) cell, write result to CSV, exit. Idempotent vs CSV."""
    from brainscore_vision import _run_score

    done = _resume_state()
    if (model_id, bench_id) in done:
        print(f"[{idx}/{total}] {model_id} {bench_id}  SKIP (cached)")
        return

    tag = f"[{idx}/{total}] {model_id:<18} {bench_id}"
    t0 = time.time()
    try:
        s = _run_score(model_id, bench_id)
        dt = time.time() - t0
        center = float(s.values)
        raw = float(s.attrs["raw"].values) if "raw" in s.attrs else float("nan")
        ceiling = float(s.attrs["ceiling"].values) if "ceiling" in s.attrs else float("nan")
        err = ""
        print(f"{tag}  score={center:+.4f}  ceil={ceiling:+.4f}  ({dt:.1f}s)")
    except Exception as e:
        dt = time.time() - t0
        center = raw = ceiling = float("nan")
        err = f"{type(e).__name__}: {e}"
        print(f"{tag}  FAIL ({dt:.1f}s): {err}")
        traceback.print_exc(limit=2)

    family, after = bench_id.split(".", 1)
    region, rest = after.split("-", 1)
    split = rest.rsplit("-", 1)[0]
    _append_row({
        "model": model_id, "benchmark": bench_id,
        "family": family, "region": region, "split": split,
        "score": center, "raw_r": raw, "ceiling": ceiling,
        "error": err, "seconds": round(dt, 1),
    })


def main() -> None:
    """Multi-cell sweep with in-process resume (kept for the original single-process use case)."""
    done = _resume_state()
    total = len(MODELS) * len(BENCHMARKS)
    print(f"Sweep matrix: {len(MODELS)} models x {len(BENCHMARKS)} benchmarks = {total} cells")
    print(f"Already done: {len(done)} -> remaining: {total - len(done)}")

    overall_t0 = time.time()
    idx = 0
    for model_id in MODELS:
        for bench_id in BENCHMARKS:
            idx += 1
            run_one(model_id, bench_id, idx=idx, total=total)

    elapsed = (time.time() - overall_t0) / 60.0
    print(f"\nSweep complete in {elapsed:.1f} min.  Long CSV: {LONG_CSV}")

    df = pd.read_csv(LONG_CSV)
    pivot = df.pivot_table(index=["family", "region", "split"], columns="model", values="score")
    pivot = pivot.reindex(columns=MODELS)
    pivot.to_csv(PIVOT_CSV)
    print(f"Pivot:    {PIVOT_CSV}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Single-cell mode: python baseline_sweep.py <model> <benchmark> [idx] [total]
        m = sys.argv[1]
        b = sys.argv[2]
        i = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        t = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        run_one(m, b, idx=i, total=t)
    else:
        main()
