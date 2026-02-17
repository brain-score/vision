from pathlib import Path

import pandas as pd

from brainscore_vision import data_registry
from brainscore_core.supported_data_standards.brainio.assemblies import (
    NeuroidAssembly, StimulusMergeAssemblyLoader,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

BIBTEX = """@article{allen_massive_2022,
    title = {A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
    volume = {25},
    issn = {1097-6256},
    doi = {10.1038/s41593-021-00962-x},
    journal = {Nature Neuroscience},
    author = {Allen, Emily J. and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L.
              and Prince, Jacob S. and Dowdle, Logan T. and Nau, Matthias and Caron, Brad
              and Pestilli, Franco and Charest, Ian and Hutchinson, J. Benjamin
              and Naselaris, Thomas and Kay, Kendrick},
    year = {2022},
    pages = {116--126},
}"""

# Local paths for pre-S3-upload development.
# Replace with load_assembly_from_s3 + proper SHA1 hashes after upload.
_LOCAL_DIR = Path("/Volumes/Hagibis/nsd/brainscore_surface")
_VOL_DIR = Path("/Volumes/Hagibis/nsd/brainscore")

# Stimulus sets are the same COCO images as the volumetric pipeline.
# They are registered by the volumetric data package (allen2022_fmri).


def _load_local_assembly(split: str, variant: str = '_8subj') -> NeuroidAssembly:
    """Load surface assembly from local netCDF + stimulus metadata.

    Used during development before S3 upload. After upload, replace the
    data_registry entries below with load_assembly_from_s3 calls.

    :param split: 'train' or 'test'
    :param variant: '_8subj' (8 subjects, 515 images) or '_4subj' (4 subjects, ~1000 images)
    """
    nc_path = _LOCAL_DIR / f"Allen2022_fmri_surface_{split}{variant}.nc"
    stim_csv = _LOCAL_DIR / f"stimulus_metadata_{split}{variant}.csv"

    meta = pd.read_csv(stim_csv)
    stimuli = StimulusSet(meta)

    # Images may reside in either stimuli_train or stimuli_test depending
    # on which variant's train/test split they belong to.
    stim_paths = {}
    for _, row in meta.iterrows():
        for d in [_VOL_DIR / "stimuli_train", _VOL_DIR / "stimuli_test"]:
            p = d / row["image_file_name"]
            if p.exists():
                stim_paths[row["stimulus_id"]] = str(p)
                break
    stimuli.stimulus_paths = stim_paths
    stimuli.identifier = f"Allen2022_fMRI_surface_{split}{variant}_Stimuli"
    stimuli.name = stimuli.identifier

    loader = StimulusMergeAssemblyLoader(
        cls=NeuroidAssembly,
        file_path=str(nc_path),
        stimulus_set_identifier=stimuli.identifier,
        stimulus_set=stimuli,
    )
    assembly = loader.load()
    assembly.attrs["identifier"] = f"Allen2022_fMRI_surface_{split}{variant}_Assembly"
    return assembly


# Default (8-subject, 515 images)
data_registry['Allen2022_fmri_surface_train'] = lambda: _load_local_assembly('train', '_8subj')
data_registry['Allen2022_fmri_surface_test'] = lambda: _load_local_assembly('test', '_8subj')

# 4-subject variant (subjects 1,2,5,7; ~1000 images)
data_registry['Allen2022_fmri_surface_4subj_train'] = lambda: _load_local_assembly('train', '_4subj')
data_registry['Allen2022_fmri_surface_4subj_test'] = lambda: _load_local_assembly('test', '_4subj')
