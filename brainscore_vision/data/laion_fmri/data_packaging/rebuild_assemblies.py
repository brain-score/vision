"""Deterministic end-to-end rebuild of the LAION-fMRI Brain-Score assemblies.

Given:
  - a configured `laion-fmri` data directory (`laion-fmri config --data-dir <path>`)
  - signed Data Use Agreement (`LAION_FMRI_REQUEST_ID` env var or
    `laion-fmri request-access` cache)

This script reproduces every artifact the benchmark needs, in order:

  1. Download all 5 subjects' GLMsingle outputs + ROIs from `s3://laion-fmri/`
     (CC0; no DUA required).
  2. Download the gated stimuli HDF5 + metadata CSV (needs DUA).
  3. Build per-subject NeuroidAssemblies (`build_assembly.py`).
  4. Filter per-subject to:
       - 1,492 shared-pool stimuli  (`prepare_shared.py`)
       - ~5,833 persubject-pool stimuli per subject (`prepare_persubject.py`)
  5. Apply the published S3 schema (`repackage_for_s3.py`) so the local files
     are byte-identical to what `load_assembly_from_s3` will pull down.
  6. Extract stimulus images from the HDF5 and package into `~/.brainio/`
     (`get_local_stimuli.py`).
  7. Semantic-verify every ``*_brainscore.nc`` output against the published
     S3 file (data values + coords element-wise). Bit-exact sha1 match is not
     possible across xarray/HDF5 versions (serialization byte-order varies),
     but data identity is enforced. Pass ``--skip-semantic-check`` to skip
     the download + comparison.

Idempotent: re-running skips any artifact already present at the expected path.

**Known divergence**: published ``shared_sub-XX_v3_brainscore.nc`` files
carry ``subject_id_pres = 'sub-XX_v3'`` (the original packaging derived
sub_id from a ``sub-XX_v3.nc`` filename stem). A fresh rebuild produces
clean ``'sub-XX'`` values. The diff is purely cosmetic -- ``subject_id_pres``
is only used as a per-subject namespace identifier and the runtime
benchmark behaves identically either way -- so the semantic verifier will
WARN on this one specific divergence per shared file. Treat 5 warnings
(one per shared subject), all reporting only the ``subject_id_pres`` coord
delta, as the expected steady state.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rebuild_assemblies")

SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
HERE = Path(__file__).parent


def run(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and stream its output. Raise on failure."""
    log.info("$ %s", " ".join(cmd))
    p = subprocess.run(cmd, **kwargs)
    if p.returncode != 0:
        raise SystemExit(f"Command failed (exit {p.returncode}): {' '.join(cmd)}")


def ensure_cc0_marker(data_dir: Path) -> None:
    """Write the CC0 license-acceptance marker so background downloads don't hang on stdin."""
    marker = data_dir / ".laion_fmri" / "license_accepted"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    log.info("Wrote CC0 marker at %s", marker)


def download_fmri(subject: str) -> None:
    run([
        "laion-fmri", "download", "--subject", subject,
        "--extension", "nii.gz", "tsv", "json",
        "--n-jobs", "4",
    ])


def download_stimuli() -> None:
    if not os.environ.get("LAION_FMRI_REQUEST_ID"):
        log.warning("LAION_FMRI_REQUEST_ID not set; the stimuli download will use the "
                    "package's cached access token (set by `laion-fmri request-access`).")
    run(["laion-fmri", "download-stimuli"])


def build_assembly(subject: str, out_dir: Path) -> Path:
    out = out_dir / f"{subject}.nc"
    if out.exists():
        log.info("[skip] %s already exists", out)
        return out
    run([sys.executable, str(HERE / "build_assembly.py"),
         "--subject", subject, "--out", str(out)])
    return out


def prepare_pool(family: str, subject: str, out_dir: Path) -> Path:
    """``family`` in {'shared', 'persubject'}."""
    out = out_dir / f"{family}_{subject}.nc"
    if out.exists():
        log.info("[skip] %s already exists", out)
        return out
    run([sys.executable, str(HERE / f"prepare_{family}.py"),
         "--in-dir", str(out_dir), "--out-dir", str(out_dir),
         "--subjects", subject])
    return out


def repackage_for_s3(out_dir: Path) -> list[Path]:
    """Run repackage_for_s3 over every shared_*.nc and persubject_*.nc in ``out_dir``.

    Outputs ``<input_stem>_brainscore.nc`` siblings -- the schema the S3
    bucket actually serves. Idempotent: each invocation only re-writes outputs
    that don't yet exist (we check first, then run only on missing siblings).
    """
    candidates: list[Path] = []
    missing: list[Path] = []
    for pattern in ("shared_sub-*.nc", "persubject_sub-*.nc"):
        for src in sorted(out_dir.glob(pattern)):
            if src.stem.endswith("_brainscore"):
                continue
            dst = out_dir / f"{src.stem}_brainscore.nc"
            candidates.append(dst)
            if not dst.exists():
                missing.append(src)

    if not missing:
        log.info("[skip] all *_brainscore.nc already exist (%d files)", len(candidates))
        return candidates

    log.info("Repackaging %d file(s) to S3-ready schema", len(missing))
    run([sys.executable, "-m",
         "brainscore_vision.data.laion_fmri.data_packaging.repackage_for_s3",
         "--in-dir", str(out_dir), "--out-dir", str(out_dir)])
    return candidates


def get_local_stimuli(stim_h5: Path, metadata_csv: Path) -> None:
    if not stim_h5.exists() or not metadata_csv.exists():
        raise FileNotFoundError(f"Stimuli HDF5 or metadata missing: {stim_h5}, {metadata_csv}")
    run([sys.executable, str(HERE / "get_local_stimuli.py"),
         "--stimuli-h5", str(stim_h5),
         "--metadata-csv", str(metadata_csv)])


def _download_published(family: str, subject: str, cache_dir: Path) -> Path:
    """Download the pinned S3 file *as a raw .nc* (bypassing the loader's reshape).

    The runtime ``load_assembly_from_s3`` returns a post-processed
    ``NeuroidAssembly`` with per-coord arrays folded into MultiIndex tuples.
    For semantic verification we want the on-disk schema, so we ``aws s3 cp``
    the versioned object directly.
    """
    from brainscore_vision.data.laion_fmri._helpers.assemblies import (
        _S3_ASSEMBLIES_SHARED, _S3_ASSEMBLIES_PERSUBJECT, _BUCKET,
    )
    manifest = {"shared": _S3_ASSEMBLIES_SHARED,
                "persubject": _S3_ASSEMBLIES_PERSUBJECT}[family]
    cfg = manifest[subject]
    bucket, *prefix_parts = _BUCKET.split("/")
    key = "/".join(prefix_parts + [f"assy_{cfg['identifier']}.nc"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"published_{family}_{subject}.nc"
    if dest.exists():
        return dest

    log.info("  downloading s3://%s/%s (version %s) -> %s",
             bucket, key, cfg["version_id"][:8] + "...", dest.name)
    subprocess.check_call([
        "aws", "s3api", "get-object",
        "--bucket", bucket, "--key", key,
        "--version-id", cfg["version_id"],
        str(dest),
    ], stdout=subprocess.DEVNULL)
    return dest


def _data_arrays_equal_chunked(a, b, rows_per_chunk: int = 1024) -> tuple[bool, int]:
    """Element-wise compare two DataArrays without allocating a full bool array.

    Returns (equal, n_diff_cells). For a (P, N) shape we iterate ``rows_per_chunk``
    presentation rows at a time, short-circuiting on first mismatch when ``equal``
    is the only thing we need. Caps peak extra RAM at ``rows_per_chunk * N * dtype``
    (~30 MB for a 1024 x 7000 float32 chunk) vs the full ~900 MB a one-shot
    ``a != b`` allocation would need.
    """
    if a.shape != b.shape:
        return False, -1
    n_diff = 0
    av = a.values
    bv = b.values
    for start in range(0, av.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, av.shape[0])
        if not np.array_equal(av[start:stop], bv[start:stop]):
            n_diff += int(np.sum(av[start:stop] != bv[start:stop]))
    return (n_diff == 0), n_diff


def _semantic_diff(rebuilt: Path, published: Path) -> list[str]:
    """Compare two on-disk .nc files structurally.

    Returns a list of human-readable mismatch descriptions; empty list = match.
    Bit-exact sha1 reproducibility is not guaranteed across xarray/HDF5
    versions, but data + coord values should be element-wise identical.

    Uses ``with`` blocks + explicit del + gc.collect to bound peak memory at
    ~2 GB (two loaded files) instead of accumulating across iterations.
    """
    import gc
    import xarray as xr
    diffs: list[str] = []

    try:
        with xr.open_dataset(rebuilt) as a_ds, xr.open_dataset(published) as b_ds:
            a = a_ds.load()
            b = b_ds.load()

            if dict(a.sizes) != dict(b.sizes):
                diffs.append(f"dims: rebuilt={dict(a.sizes)}  published={dict(b.sizes)}")

            a_data = a.get("data") if "data" in a.data_vars else next(iter(a.data_vars.values()))
            b_data = b.get("data") if "data" in b.data_vars else next(iter(b.data_vars.values()))
            if a_data.shape != b_data.shape:
                diffs.append(f"data shape: rebuilt={a_data.shape}  published={b_data.shape}")
            else:
                equal, n_cells = _data_arrays_equal_chunked(a_data, b_data)
                if not equal:
                    diffs.append(f"data: {n_cells} cells differ")

            rebuilt_coords = set(a.coords)
            published_coords = set(b.coords)
            if rebuilt_coords != published_coords:
                only_new = rebuilt_coords - published_coords
                only_old = published_coords - rebuilt_coords
                if only_new:
                    diffs.append(f"coord-set: only in rebuilt = {sorted(only_new)}")
                if only_old:
                    diffs.append(f"coord-set: only in published = {sorted(only_old)}")

            for c in sorted(rebuilt_coords & published_coords):
                av = a[c].values
                bv = b[c].values
                if av.shape != bv.shape:
                    diffs.append(f"coord {c!r}: shape rebuilt={av.shape}  published={bv.shape}")
                    continue
                if av.dtype.kind in "iufb":
                    equal = np.array_equal(av, bv)
                else:
                    equal = bool((av == bv).all())
                if not equal:
                    sample = (str(av[:3]), str(bv[:3])) if av.ndim == 1 else ("...", "...")
                    diffs.append(
                        f"coord {c!r}: differs  rebuilt[:3]={sample[0]}  published[:3]={sample[1]}"
                    )
    finally:
        # Hard-release the loaded arrays before the caller's next iteration.
        a = b = a_ds = b_ds = a_data = b_data = None  # noqa: F841
        gc.collect()

    return diffs


def verify_semantic(out_dir: Path, subjects: list[str], keep_cache: bool = False) -> int:
    """Compare every rebuilt ``*_brainscore.nc`` against its pinned S3 counterpart.

    Returns the number of files with semantic differences. Logs each diff
    inline. Note: the *first* discrepancy this catches is a real one to
    investigate -- cosmetic byte-level differences (timestamps, attribute
    order) are invisible to this check by design.

    Downloaded published files are deleted immediately after the per-file
    comparison to keep disk usage bounded at roughly one file (~1 GB) instead
    of summing across all 10 (~5 GB). Pass ``keep_cache=True`` to leave the
    downloaded files in place (e.g. for iterative debugging of one diff).
    """
    n_diff = 0
    cache_dir = out_dir / ".verify_cache"
    log.info("Semantic-verifying %d files against published S3 artifacts",
             len(subjects) * 2)
    for family in ("shared", "persubject"):
        for sub in subjects:
            local = out_dir / f"{family}_{sub}_brainscore.nc"
            if not local.exists():
                log.error("MISSING: %s", local)
                n_diff += 1
                continue
            published = _download_published(family, sub, cache_dir)
            try:
                diffs = _semantic_diff(local, published)
            finally:
                if not keep_cache and published.exists():
                    published.unlink()
            if not diffs:
                log.info("  [%s] %-8s  OK (data + coords match published)", family, sub)
            else:
                log.warning("  [%s] %-8s  DIFFERS:", family, sub)
                for d in diffs:
                    log.warning("      - %s", d)
                n_diff += 1

    # Clean up the cache directory itself if we own it and it's empty.
    if not keep_cache and cache_dir.exists():
        try:
            cache_dir.rmdir()
        except OSError:
            pass  # not empty -- leave it

    if n_diff == 0:
        log.info("All %d files semantically match published artifacts.",
                 len(subjects) * 2)
    else:
        log.warning("%d file(s) differ semantically from published. "
                    "Investigate before considering the rebuild canonical.",
                    n_diff)
    return n_diff


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path.home() / "laion-fmri",
                   help="LAION-fMRI data directory (matches `laion-fmri config`). "
                        "Default: ~/laion-fmri/")
    p.add_argument("--out-dir", type=Path,
                   default=Path.home() / "laion-fmri/assemblies",
                   help="Where to write per-subject .nc files. "
                        "Default: ~/laion-fmri/assemblies/")
    p.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    p.add_argument("--skip-stimuli", action="store_true",
                   help="Skip the gated stimuli download + local-cache packaging.")
    p.add_argument("--skip-semantic-check", action="store_true",
                   help="Skip the post-rebuild semantic comparison against published S3 artifacts.")
    p.add_argument("--request-id", default=None,
                   help="LAION-fMRI Data Use Agreement request ID (from "
                        "https://laion-fmri.hebartlab.com/request). Sets "
                        "LAION_FMRI_REQUEST_ID for the stimulus download step. "
                        "If omitted, the existing env var or the package's "
                        "cached access token is used.")
    args = p.parse_args()

    if args.request_id:
        os.environ["LAION_FMRI_REQUEST_ID"] = args.request_id

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_cc0_marker(args.data_dir)

    log.info("== STEP 1: fMRI download (CC0) ==")
    for sub in args.subjects:
        download_fmri(sub)

    log.info("== STEP 2: per-subject assembly build ==")
    for sub in args.subjects:
        build_assembly(sub, args.out_dir)

    log.info("== STEP 3a: shared-pool slicing ==")
    for sub in args.subjects:
        prepare_pool("shared", sub, args.out_dir)

    log.info("== STEP 3b: persubject-pool slicing ==")
    for sub in args.subjects:
        prepare_pool("persubject", sub, args.out_dir)

    log.info("== STEP 4: repackage to S3 schema ==")
    repackage_for_s3(args.out_dir)

    if not args.skip_stimuli:
        log.info("== STEP 5: stimuli download + local cache (DUA) ==")
        download_stimuli()
        stim_h5 = args.data_dir / "stimuli" / "task-images_stimuli.h5"
        metadata_csv = args.data_dir / "stimuli" / "task-images_metadata.csv"
        get_local_stimuli(stim_h5, metadata_csv)
    else:
        log.info("== STEP 5: skipped (--skip-stimuli) ==")

    if not args.skip_semantic_check:
        log.info("== STEP 6: semantic verification against published S3 ==")
        n_diff = verify_semantic(args.out_dir, args.subjects)
        if n_diff:
            log.warning("Rebuild complete with %d semantic difference(s) -- "
                        "see logs above. PR-readiness depends on whether the "
                        "differences are intentional.", n_diff)
    else:
        log.info("== STEP 6: skipped (--skip-semantic-check) ==")

    log.info("Done.")


if __name__ == "__main__":
    main()
