"""Repackage per-subject .nc files into Brain-Score S3-ready format.

The only transform is **single-data_var collapse**:
``xr.DataArray.to_netcdf`` (xarray 2022.x) serialises non-dimensional coords
as data_vars, which makes
``brainscore_core.brainio.s3.load_assembly_from_s3`` fail
("Given file dataset contains more than one data variable"). Brain-Score's
convention is a single ``data`` data_var with everything else as proper
coords. We reconstruct the DataArray to enforce that.

That's it. The input .nc files (``shared_sub-XX.nc`` from
``prepare_shared.py``, ``persubject_sub-XX.nc`` from
``prepare_persubject.py``) already carry whatever presentation-side schema
the published S3 artifacts need. In particular:

  - ``prepare_shared.py`` strips ``session``/``run``/``beta_index``,
    renumbers ``repetition`` within stimulus, and adds ``subject_id_pres`` as
    a presentation coord -- matching the existing
    ``shared_sub-XX_v3_brainscore.nc`` schema on S3.
  - ``prepare_persubject.py`` keeps the bookkeeping coords and lets
    ``make_persubject_loader`` perform load-time surgery -- matching the
    existing ``persubject_sub-XX_v3_brainscore.nc`` schema on S3 (and the
    backward-compat path acknowledged as tech debt in
    ``_helpers/assemblies.py``).

If you want to migrate to a cleaner persubject schema, add the schema fixes
to ``prepare_persubject.py``, re-run the rebuild, capture new sha1s, update
the manifest in ``_helpers/assemblies.py``, and re-upload to S3.

Run via:
    # All sub-*.nc and shared_sub-*.nc / persubject_sub-*.nc in --in-dir:
    python -m brainscore_vision.data.laion_fmri.data_packaging.repackage_for_s3

Output naming convention: ``<input_stem>_brainscore.nc`` (one per input).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import xarray as xr


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("repackage")


def _to_single_data_var(da: xr.DataArray) -> xr.DataArray:
    """Reconstruct as a single ``data`` DataArray so to_netcdf produces one data_var."""
    return xr.DataArray(
        da.values,
        dims=da.dims,
        coords=dict(da.coords),
        name="data",
    )


def repackage(in_path: Path, out_path: Path) -> None:
    log.info("repackage %s -> %s", in_path.name, out_path.name)
    ds = xr.open_dataset(in_path).load()
    if "data" in ds.data_vars and ds["data"].dims == ("presentation", "neuroid"):
        da = ds["data"]
    else:
        candidates = [v for v in ds.data_vars
                      if ds[v].dims == ("presentation", "neuroid")]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{in_path.name}: expected one (presentation, neuroid) data_var, "
                f"got {candidates}"
            )
        da = ds[candidates[0]]
        # Reattach the other variables as coords.
        for var in ds.data_vars:
            if var == candidates[0]:
                continue
            da = da.assign_coords({var: ds[var]})
        for c in ds.coords:
            if c not in da.coords:
                da = da.assign_coords({c: ds[c]})

    log.info("  shape: %s   coords: %d", da.shape, len(da.coords))
    da = _to_single_data_var(da)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out_path)
    log.info("  wrote %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Sanity round-trip: must be exactly one data_var named `data`.
    ds2 = xr.open_dataset(out_path)
    assert list(ds2.data_vars) == ["data"], (
        f"expected single 'data' data_var, got {list(ds2.data_vars)}"
    )


def main():
    default_dir = Path.home() / "laion-fmri/assemblies"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, default=default_dir,
                   help="Where input .nc files live. Globs for shared_sub-*.nc, "
                        "persubject_sub-*.nc, and sub-*.nc (excluding outputs). "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--out-dir", type=Path, default=default_dir,
                   help="Where to write the *_brainscore.nc outputs. "
                        "Default: ~/laion-fmri/assemblies/")
    args = p.parse_args()

    patterns = ("shared_sub-*.nc", "persubject_sub-*.nc", "sub-*.nc")
    seen: set[Path] = set()
    for pattern in patterns:
        for src in sorted(args.in_dir.glob(pattern)):
            if src.stem.endswith("_brainscore") or src in seen:
                continue
            seen.add(src)
            dst = args.out_dir / f"{src.stem}_brainscore.nc"
            repackage(src, dst)

    if not seen:
        log.warning("No matching input files found under %s", args.in_dir)


if __name__ == "__main__":
    main()
