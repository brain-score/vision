"""Per-subject assembly loader factories.

The LAION-fMRI release ships ten S3-stored ``.nc`` assemblies: 5 subjects ×
2 stim-pool families (shared 1,492 stim, persubject 6,204 stim incl. OOD).
Rather than
write ten near-identical lambdas in ``__init__.py``, we expose a factory that
takes a subject ID and returns a loader closure pinned to that subject's
S3 version_id + sha1.

Two factories live here, one per family:

- :func:`make_shared_loader(sub_id)` — straight ``load_assembly_from_s3`` call.
- :func:`make_persubject_loader(sub_id)` — the persubject ``.nc`` files were
  uploaded with trial-level metadata (``session``/``run``/``beta_index``) on the
  presentation MultiIndex and without ``subject_id_pres``. Post-load surgery
  strips the trial-level coords (so ``average_repetition`` actually averages
  rather than producing one group per trial) and adds ``subject_id_pres`` as a
  MultiIndex level (so it survives ``filter_reliable_neuroids``'s
  ``where(drop=True)`` and namespaces the activations cache per subject).

  This reshape is tech debt — repackaging the assemblies with the correct
  schema would let this factory mirror :func:`make_shared_loader`. Tracked
  but deferred.
"""

from __future__ import annotations

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3
from brainscore_vision import load_stimulus_set


_BUCKET = "brainscore-storage/brainscore-vision/benchmarks/LAION_fMRI"


# Uploaded 2026-05-18 (v3 — NaN-fixed z-scoring + single `data` data_var, metadata as coords).
_S3_ASSEMBLIES_SHARED = {
    "sub-01": dict(identifier="LAION_fMRI_full_sub-01_Assembly",
                   version_id="ZN0QBmAbna8F1mqabGMEWNlhSHqclRkj",
                   sha1="5d928ad1b170fa2fa58bcf1215ba60c4dd389ef4"),
    "sub-03": dict(identifier="LAION_fMRI_full_sub-03_Assembly",
                   version_id="vq6BErgrj2e6SMQdfxfwh0XVLd.FsxCX",
                   sha1="ecff27ddc4dc0a75cfbcc719b91c3739c1ba2bb1"),
    "sub-05": dict(identifier="LAION_fMRI_full_sub-05_Assembly",
                   version_id="Tzl5F826dIUNG7jGY8soR07zJ0DC3ffZ",
                   sha1="c446c8a6c67de21e749469a092fd02994d2be01e"),
    "sub-06": dict(identifier="LAION_fMRI_full_sub-06_Assembly",
                   version_id="2FaJ3bLhgR_XDDERM38RwzTFfc8e3YqM",
                   sha1="912aec9934c7b2daafb4d3927db895363b9eb0c1"),
    "sub-07": dict(identifier="LAION_fMRI_full_sub-07_Assembly",
                   version_id="HGT8RP0IbkFFeG7KI.oifhOSm3YlD1yD",
                   sha1="e16a456229427eb876b7ee85e0019570130fc4f0"),
}

# Per-subject pool variants (1,121 shared non-OOD + 4,712 subject-unique +
# 371 OOD = 6,204 unique stimuli per subject; ~31.8K trial-level rows).
# 4× more unique stimuli, ~2.3× tighter per-voxel correlation CIs than shared.
_S3_ASSEMBLIES_PERSUBJECT = {
    "sub-01": dict(identifier="LAION_fMRI_persubject_sub-01_Assembly",
                   version_id="ZMebfDS6bT7DjDqKcuHmOu9DlRbPstk6",
                   sha1="827781fe183ee1744f968517c1ea5afbe7860d4b"),
    "sub-03": dict(identifier="LAION_fMRI_persubject_sub-03_Assembly",
                   version_id="KFwyIrU_1hqbPY64iKxE1CWvAHRmLaL2",
                   sha1="f699b3aeb168a5a6f57a48704bdb03b0103d34d3"),
    "sub-05": dict(identifier="LAION_fMRI_persubject_sub-05_Assembly",
                   version_id="As0uxDjHTzjtnGSjEVxBqnIEdK0zHq0d",
                   sha1="73a1ee4f85c41c1c30f771a46396a91c73a85b6c"),
    "sub-06": dict(identifier="LAION_fMRI_persubject_sub-06_Assembly",
                   version_id="rCLfMcMoYoyeVu.6meVUzfNEzWkfV8yJ",
                   sha1="1d7a43878a05787c048696c023575850e0b2ba3b"),
    "sub-07": dict(identifier="LAION_fMRI_persubject_sub-07_Assembly",
                   version_id="z1fAn.bLXyrw_PSoFz458m19hN8vOfW_",
                   sha1="5c2d025d41f4befa449eca2b390dff1ab2f29821"),
}


def make_shared_loader(sub_id: str):
    """Return a ``load_dataset``-style closure for the shared-pool subject assembly."""
    cfg = _S3_ASSEMBLIES_SHARED[sub_id]

    def _load():
        return load_assembly_from_s3(
            identifier=cfg["identifier"],
            version_id=cfg["version_id"],
            sha1=cfg["sha1"],
            bucket=_BUCKET,
            cls=NeuroidAssembly,
            stimulus_set_loader=lambda: load_stimulus_set("Zerbe2026_fmri_stim_full"),
            # Presentation dim already carries `stimulus_id`; merging stim_set
            # metadata (image dimensions, etc.) into the assembly isn't needed.
            merge_stimulus_set_meta=False,
        )

    return _load


def make_persubject_loader(sub_id: str):
    """Return a loader closure that post-processes the persubject assembly.

    See module docstring for the schema-fixup rationale.
    """
    cfg = _S3_ASSEMBLIES_PERSUBJECT[sub_id]

    def _load():
        da = load_assembly_from_s3(
            identifier=cfg["identifier"], version_id=cfg["version_id"],
            sha1=cfg["sha1"], bucket=_BUCKET, cls=NeuroidAssembly,
            stimulus_set_loader=lambda: load_stimulus_set("Zerbe2026_fmri_stim_full"),
            merge_stimulus_set_meta=False,
        )
        # 1. Strip session/run/beta_index so `average_repetition` actually averages.
        # 2. Add subject_id_pres as a MultiIndex level so it survives
        #    `where(drop=True)` and namespaces the activations cache per subject.
        trial_levels = ("session", "run", "beta_index")
        present_levels = (
            da.indexes["presentation"].names if "presentation" in da.indexes else ()
        )
        kept_levels = [lv for lv in present_levels
                       if lv not in trial_levels and lv != "presentation_level_0"]
        da = da.reset_index("presentation")
        for lv in trial_levels:
            if lv in da.coords:
                da = da.drop_vars(lv, errors="ignore")
        if "presentation_level_0" in da.coords:
            da = da.drop_vars("presentation_level_0", errors="ignore")
        # Inject subject_id_pres BEFORE building the MultiIndex so set_index
        # picks it up as a level (not a regular coord).
        n_pres = da.sizes["presentation"]
        da = da.assign_coords(
            subject_id_pres=("presentation", np.full(n_pres, sub_id, dtype=object))
        )
        kept_levels = kept_levels + ["subject_id_pres"]
        da = da.set_index(presentation=kept_levels)
        return da

    return _load
