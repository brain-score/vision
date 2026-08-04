"""Per-subject shared-pool preparation for the LAION-fMRI benchmark.

Replaces the earlier cross-subject `concat_shared.py` (which materialized a 5+ GB
NaN-padded matrix in RAM and OOM'd). For the Allen2022-style shared-pool benchmark
we don't need a single cross-subject assembly — `TrainTestNeuralBenchmark` with
`alpha_coord='subject'` slices per-subject anyway. So we ship per-subject .nc files
and let the benchmark loader concatenate small filtered slices at eval time.

Per subject:
  1. Load the full sub-XX.nc built by build_assembly.py.
  2. Filter presentations to the 1,492 shared-pool stimulus_ids.
  3. Drop trial-bookkeeping coords (session, run, beta_index) so multi_groupby
     averaging in `average_repetition` collapses on (stimulus_id, subject_id, ...).
  4. Re-label `repetition` 0..N-1 within (stimulus_id) within this subject.
  5. Promote `subject_id` to BOTH presentation and neuroid coords.
  6. Write `shared_<sub>.nc` — typically ~370 MB, vs 905 MB for the full file.

Output filenames: `<out-dir>/shared_<sub_id>.nc`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_shared")


def shared_pool_stimulus_ids() -> set[str]:
    """Return all stimulus_ids in the shared pool (1,121 non-OOD + 371 OOD = 1,492)."""
    from laion_fmri.splits import get_train_test_ids

    tau_train, tau_test = get_train_test_ids("tau", pool="shared")
    _, ood_test = get_train_test_ids("ood", pool="shared")
    return set(tau_train) | set(tau_test) | set(ood_test)


def prepare_subject(in_path: Path, out_path: Path, shared_ids: set[str]) -> None:
    """Filter, normalize, and save one subject's shared-pool slice."""
    sub_id = in_path.stem
    log.info("Loading %s", in_path)
    da = xr.open_dataarray(in_path).load()
    n_full = da.sizes["presentation"]

    keep = pd.Series(da["stimulus_id"].values).isin(shared_ids).to_numpy()
    da = da.isel(presentation=np.flatnonzero(keep))
    log.info("  %s: %d / %d trials kept in shared pool", sub_id, da.sizes["presentation"], n_full)

    drop = [c for c in ("session", "run", "beta_index") if c in da.coords]
    if drop:
        da = da.drop_vars(drop)
        log.info("  dropped presentation coords: %s", drop)

    # Re-number repetition 0..N-1 within stimulus_id (within this subject's slice).
    new_rep = (
        pd.DataFrame({"sid": da["stimulus_id"].values})
        .groupby("sid")
        .cumcount()
        .to_numpy(dtype=np.int32)
    )
    if "repetition" in da.coords:
        da = da.drop_vars("repetition")
    da = da.assign_coords(repetition=("presentation", new_rep))

    # Add presentation-side subject coord; the existing neuroid-side subject_id stays.
    # NOTE: the originally-published shared_sub-XX_v3_brainscore.nc files have
    # `subject_id_pres = 'sub-XX_v3'` because the original packaging run derived
    # sub_id from a `sub-XX_v3.nc` filename stem, baking the version suffix into
    # the values. We deliberately produce clean 'sub-XX' here -- the value is
    # only used as a per-subject namespace identifier, so the difference is
    # cosmetic. `rebuild_assemblies.py`'s semantic verifier flags this as a
    # known/expected divergence vs the S3 artifacts.
    sid_pres = np.full(da.sizes["presentation"], sub_id, dtype=object)
    da = da.assign_coords(subject_id_pres=("presentation", sid_pres))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out_path)
    log.info("  wrote %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main():
    # Default layout matches the runtime stimulus loader's expected location;
    # override either with --in-dir / --out-dir if your data lives elsewhere.
    default_dir = Path.home() / "laion-fmri/assemblies"
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=default_dir,
                   help="Directory holding the per-subject sub-XX.nc assemblies. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--out-dir", type=Path, default=default_dir,
                   help="Where to write the shared-pool variant. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Subject IDs to process (default: every sub-*.nc in --in-dir).")
    args = p.parse_args()

    if args.subjects:
        paths = [args.in_dir / f"{s}.nc" for s in args.subjects]
    else:
        paths = sorted(args.in_dir.glob("sub-*.nc"))
    log.info("Processing %d subject(s)", len(paths))

    shared_ids = shared_pool_stimulus_ids()
    log.info("Shared pool: %d unique stimulus_ids", len(shared_ids))

    for in_path in paths:
        out_path = args.out_dir / f"shared_{in_path.stem}.nc"
        prepare_subject(in_path, out_path, shared_ids)


if __name__ == "__main__":
    main()
