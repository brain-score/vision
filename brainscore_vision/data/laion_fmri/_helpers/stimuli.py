"""DUA-gated stimulus loading for LAION-fMRI.

The LAION-fMRI Data Use Agreement prohibits redistribution of the image bytes.
Users obtain stimuli locally via the upstream ``laion-fmri request-access``
(signs DUA) + ``laion-fmri download-stimuli`` CLI, then run
``data_packaging/get_local_stimuli.py`` to extract JPEGs and write a manifest.

Brain-Score also holds a private S3 mirror at
``s3://brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI/stimuli/``
that's reachable only via authenticated ``aws s3 cp`` — used as a fallback so
Brain-Score's evaluator hosts (which have AWS creds) can fetch on first use.

This module exists separately from ``__init__.py`` because the standard
``brainio.s3.load_stimulus_set_from_s3`` (anonymous boto3) can't be used for
DUA-gated data.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


_STIMULI_S3_BUCKET = "brainscore-storage"
_STIMULI_S3_KEY = "brainscore-vision/benchmarks/Zerbe2026_fmri/stimuli/images_extracted.zip"


def _stimuli_dir_candidates() -> list[Path]:
    """Local directories to probe (in priority order) for an extracted stim set.

    Override with the ``LAION_FMRI_STIMULI_DIR`` environment variable. Otherwise
    we look in the canonical layout produced by
    ``data_packaging/get_local_stimuli.py`` — i.e. ``~/laion-fmri/stimuli/images_extracted``
    containing the JPEGs + ``manifest.csv``.
    """
    out: list[Path] = []
    env = os.environ.get("LAION_FMRI_STIMULI_DIR")
    if env:
        out.append(Path(env))
    out.append(Path.home() / "laion-fmri/stimuli/images_extracted")
    return out


def _resolve_local_stimuli_dir() -> Path:
    """Return a local directory containing ``manifest.csv`` + JPEGs.

    Probes the candidate locations first; if none carry the manifest, falls
    back to fetching the private S3 zip (~3.2 GB) into ``~/laion-fmri/`` once.
    """
    for candidate in _stimuli_dir_candidates():
        if (candidate / "manifest.csv").exists():
            return candidate

    target_dir = Path.home() / "laion-fmri/stimuli/images_extracted"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir.parent / "images_extracted.zip"
    if not zip_path.exists():
        subprocess.run(
            ["aws", "s3", "cp",
             f"s3://{_STIMULI_S3_BUCKET}/{_STIMULI_S3_KEY}", str(zip_path)],
            check=True,
        )
    if not (target_dir / "manifest.csv").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir.parent)  # zip contains an `images_extracted/` folder
    return target_dir


def load_stimulus_set() -> StimulusSet:
    """Build a Brain-Score StimulusSet from the locally-cached LAION-fMRI images.

    The returned StimulusSet carries identifiers + metadata plus a
    ``stimulus_paths`` dict so ``candidate.look_at()`` resolves JPEGs locally.
    Rewrites the manifest's ``path`` column per-row to the current host's stim
    directory, so a manifest written on one machine still works after caching
    on another.
    """
    stim_dir = _resolve_local_stimuli_dir()
    manifest = pd.read_csv(stim_dir / "manifest.csv")
    manifest["path"] = manifest["filename"].apply(lambda f: str(stim_dir / f))
    stim = StimulusSet(manifest.drop(columns=["path"]))
    stim.identifier = "Zerbe2026_fmri_stim_full"
    stim.stimulus_paths = dict(zip(manifest["stimulus_id"], manifest["path"]))
    return stim
