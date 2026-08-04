"""LAION-fMRI local stimulus extraction.

The stimulus images are gated by a Data Use Agreement and cannot be redistributed
via Brain-Score's public S3 bucket. This script extracts the user's DUA-approved
``task-images_stimuli.h5`` archive into a flat directory of JPEGs plus a manifest CSV,
ready for the benchmark's stimulus-set loader to consume locally.

**Full setup walkthrough** (DUA → download → this script → first score) lives in
``vision/brainscore_vision/benchmarks/laion_fmri/README.md`` § "Stimulus setup".

Quick invocation (after running ``laion-fmri download-stimuli``)::

    python -m brainscore_vision.data.laion_fmri.data_packaging.get_local_stimuli \\
        --stimuli-h5    ~/laion-fmri/stimuli/task-images_stimuli.h5 \\
        --metadata-csv  ~/laion-fmri/stimuli/task-images_metadata.csv

Output (defaults to ``~/laion-fmri/stimuli/images_extracted/``)::

    <out-dir>/                          25,052 JPEG files, named after their stimulus_id
    <out-dir>/manifest.csv              one row per image: stimulus_id, filename, metadata

The runtime stimulus-set loader (``data/laion_fmri/_helpers/stimuli.py``) probes
this layout by default; override with the ``LAION_FMRI_STIMULI_DIR`` env var.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("get_local_stimuli")


def extract_jpegs(stimuli_h5: Path, metadata_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Extract every JPEG from the gated HDF5 archive into `out_dir`.

    The HDF5 has a single dataset `images` of shape (25052,) and object dtype; each
    item is a uint8 ndarray of JPEG-encoded bytes. The metadata CSV is row-aligned
    with `images` — CSV row N -> HDF5 image N — and provides the canonical
    `image_name` (which equals the `label` column in the per-session trial-info TSVs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(metadata_csv)
    if "image_name" not in meta.columns:
        raise RuntimeError(
            f"Expected `image_name` column in {metadata_csv}, found: {list(meta.columns)}"
        )

    records: list[dict] = []
    with h5py.File(stimuli_h5, "r") as f:
        if "images" not in f:
            raise RuntimeError(
                f"Expected `images` dataset in {stimuli_h5}, found: {list(f.keys())}"
            )
        images = f["images"]
        if len(images) != len(meta):
            raise RuntimeError(
                f"HDF5 has {len(images)} images but metadata has {len(meta)} rows; "
                "the dataset assumes row-alignment."
            )

        log.info("Extracting %d JPEGs to %s", len(images), out_dir)
        for i, image_id in enumerate(meta["image_name"].values):
            img_bytes = bytes(np.asarray(images[i], dtype=np.uint8))

            # stimulus_id values like `shared_12rep_LAION_cluster_1003_i0.jpg` are
            # filename-safe; preserve as-is so `label` in trial_info matches exactly.
            out_path = out_dir / image_id
            out_path.write_bytes(img_bytes)

            records.append({
                "stimulus_id": image_id,
                "filename": image_id,
                "path": str(out_path),
            })

            if i and i % 2000 == 0:
                log.info("  %d / %d", i, len(images))

    manifest = pd.DataFrame.from_records(records).merge(
        meta.rename(columns={"image_name": "stimulus_id"}),
        on="stimulus_id", how="left", validate="one_to_one",
    )
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stimuli-h5", required=True, type=Path,
                   help="Path to task-images_stimuli.h5 (DUA-gated HDF5 archive).")
    p.add_argument("--metadata-csv", required=True, type=Path,
                   help="Path to task-images_metadata.csv.")
    p.add_argument("--out-dir", type=Path,
                   default=Path.home() / "laion-fmri/stimuli/images_extracted",
                   help="Directory to write per-image JPEGs and manifest.csv. "
                        "Default matches the layout the runtime stimulus loader probes "
                        "(~/laion-fmri/stimuli/images_extracted/).")
    args = p.parse_args()

    if not args.stimuli_h5.exists():
        raise SystemExit(f"Missing stimuli HDF5: {args.stimuli_h5}")
    if not args.metadata_csv.exists():
        raise SystemExit(f"Missing metadata CSV: {args.metadata_csv}")

    manifest = extract_jpegs(args.stimuli_h5, args.metadata_csv, args.out_dir)
    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log.info("Wrote manifest: %s  (%d rows)", manifest_path, len(manifest))

    # Quick sanity stats
    by_dataset = manifest["dataset"].value_counts().to_dict()
    by_shared = manifest["unique_or_shared"].value_counts().to_dict()
    log.info("By dataset: %s", by_dataset)
    log.info("By unique_or_shared: %s", by_shared)


if __name__ == "__main__":
    main()
