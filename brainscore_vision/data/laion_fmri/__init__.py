"""LAION-fMRI data registry.

Five subjects × two stim-pool families = 10 assembly identifiers, plus one
stimulus set registered under both family names (since ``benchmark.py``
derives the stim-set identifier from the dataset prefix).

Loader factories + DUA-gated stimulus handling live in ``_helpers/`` to keep
this file grep-friendly: Brain-Score's plugin discovery text-matches
``data_registry['<identifier>']`` literals in plugin ``__init__.py`` files,
so the registry assignments must be explicit and live here.
"""

from brainscore_vision import data_registry, stimulus_set_registry
from ._helpers import load_stimulus_set, make_persubject_loader, make_shared_loader


BIBTEX = """@inproceedings{zerbe_laion-fmri_2026,
    title = {{LAION}-{fMRI}: A densely sampled 7T-fMRI dataset providing broad coverage of natural image diversity},
    author = {Zerbe, Josefine and Roth, Johannes and Mell, Maggie Mae and Herholz, Peer and Knapen, Tomas and Hebart, Martin N.},
    year = {2026},
    booktitle = {Vision Sciences Society Annual Meeting},
}"""

SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


# ── Shared-pool assemblies (1,492 stim seen by every subject) ─────────────
data_registry['Zerbe2026_fmri_full_sub-01'] = make_shared_loader('sub-01')
data_registry['Zerbe2026_fmri_full_sub-03'] = make_shared_loader('sub-03')
data_registry['Zerbe2026_fmri_full_sub-05'] = make_shared_loader('sub-05')
data_registry['Zerbe2026_fmri_full_sub-06'] = make_shared_loader('sub-06')
data_registry['Zerbe2026_fmri_full_sub-07'] = make_shared_loader('sub-07')

# ── Per-subject pool assemblies (5,833 stim/subj: 1,121 shared + 4,712 unique) ──
data_registry['Zerbe2026_fmri_persubject_full_sub-01'] = make_persubject_loader('sub-01')
data_registry['Zerbe2026_fmri_persubject_full_sub-03'] = make_persubject_loader('sub-03')
data_registry['Zerbe2026_fmri_persubject_full_sub-05'] = make_persubject_loader('sub-05')
data_registry['Zerbe2026_fmri_persubject_full_sub-06'] = make_persubject_loader('sub-06')
data_registry['Zerbe2026_fmri_persubject_full_sub-07'] = make_persubject_loader('sub-07')

# ── Stimulus set (DUA-gated; loader probes local paths then private S3) ──
# Registered under both family names because benchmark.py derives the
# stim-set identifier from `f"{dataset_prefix}_stim_full"`.
stimulus_set_registry['Zerbe2026_fmri_stim_full'] = load_stimulus_set
stimulus_set_registry['Zerbe2026_fmri_persubject_stim_full'] = load_stimulus_set
