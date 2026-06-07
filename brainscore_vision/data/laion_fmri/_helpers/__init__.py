"""Loader factories used by the LAION-fMRI data registry.

Kept in a subfolder so the main ``data/laion_fmri/__init__.py`` stays a
simple, grep-friendly registry.
"""

from .assemblies import make_persubject_loader, make_shared_loader
from .stimuli import load_stimulus_set

__all__ = ["load_stimulus_set", "make_persubject_loader", "make_shared_loader"]
