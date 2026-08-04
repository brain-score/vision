"""Per-subject persubject-pool preparation for the LAION-fMRI benchmark.

Twin of :mod:`prepare_shared` for the *persubject* stim pool (each subject's
own ~6,204 stimuli: 1,121 shared non-OOD + 4,712 subject-unique + 371 OOD).
The persubject pool drives the headline ridge benchmark (see
``benchmarks/laion_fmri/METHODS.md`` ?? "Difficulty relative to Allen2022 /
NSD").

Per subject:
  1. Load the full ``sub-XX.nc`` built by ``build_assembly.py``.
  2. Filter presentations to the stim_ids that belong to *this subject's*
     persubject pool (``laion_fmri.splits.get_train_test_ids(pool=sub_id)``).
  3. Write ``persubject_<sub>.nc`` unchanged otherwise -- trial-bookkeeping
     coords (session, run, beta_index) stay on the file and are stripped at
     load time by ``make_persubject_loader``. We keep the on-disk schema
     minimal here because the existing persubject S3 artifacts were produced
     this way; the load-time surgery is intentionally preserved as
     backward-compat for any cached older copies (acknowledged tech debt in
     ``_helpers/assemblies.py``).

Output filenames: ``<out-dir>/persubject_<sub_id>.nc``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_persubject")


def persubject_pool_stimulus_ids(sub_id: str) -> set[str]:
    """Return all stimulus_ids in ``sub_id``'s persubject pool.

    Union of tau train + tau test + ood test for ``pool=sub_id``. Same
    convention as the runtime split loader; staying consistent ensures the
    packaged assembly carries every stim the benchmark will ever ask for.
    """
    from laion_fmri.splits import get_train_test_ids

    tau_train, tau_test = get_train_test_ids("tau", pool=sub_id)
    _, ood_test = get_train_test_ids("ood", pool=sub_id)
    return set(tau_train) | set(tau_test) | set(ood_test)


def prepare_subject(in_path: Path, out_path: Path) -> None:
    """Filter and save one subject's persubject-pool slice (no schema fixes)."""
    sub_id = in_path.stem
    log.info("Loading %s", in_path)
    da = xr.open_dataarray(in_path).load()
    n_full = da.sizes["presentation"]

    pool_ids = persubject_pool_stimulus_ids(sub_id)
    log.info("  %s persubject pool: %d unique stimulus_ids", sub_id, len(pool_ids))

    import pandas as pd
    keep = pd.Series(da["stimulus_id"].values).isin(pool_ids).to_numpy()
    da = da.isel(presentation=np.flatnonzero(keep))
    log.info("  %s: %d / %d trials kept in persubject pool",
             sub_id, da.sizes["presentation"], n_full)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out_path)
    log.info("  wrote %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main():
    default_dir = Path.home() / "laion-fmri/assemblies"
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=default_dir,
                   help="Directory holding the per-subject sub-XX.nc assemblies. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--out-dir", type=Path, default=default_dir,
                   help="Where to write the persubject-pool variant. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Subject IDs to process (default: every sub-*.nc in --in-dir).")
    args = p.parse_args()

    if args.subjects:
        paths = [args.in_dir / f"{s}.nc" for s in args.subjects]
    else:
        paths = sorted(args.in_dir.glob("sub-*.nc"))
    log.info("Processing %d subject(s)", len(paths))

    for in_path in paths:
        out_path = args.out_dir / f"persubject_{in_path.stem}.nc"
        prepare_subject(in_path, out_path)


if __name__ == "__main__":
    main()
