from pathlib import Path

import pandas as pd

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_stimulus_set_from_s3, load_assembly_from_s3,
)
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

# Stimulus sets
stimulus_set_registry['Allen2022_fmri_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_train_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
    csv_sha1="b30e2ed9730c905b9b8673155230b762571e71d5",
    zip_sha1="8aba5aefed91ea153451ddcba79c08eece627ed2",
    csv_version_id="null",
    zip_version_id="null",
    filename_prefix="stimulus_")

stimulus_set_registry['Allen2022_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
    csv_sha1="d5d39fa5fa492c0249b075bdc7e2a514b239accf",
    zip_sha1="78276b6a4f32261f211ce4d3fa51294e8f14d30b",
    csv_version_id="null",
    zip_version_id="null",
    filename_prefix="stimulus_")

# Local path for pre-S3-upload development (streams-ventral IT ROI, dual variants).
# Replace with load_assembly_from_s3 + proper SHA1 hashes after upload.
_LOCAL_DIR = Path("/Volumes/Hagibis/nsd/brainscore")


def _load_local_assembly(split: str, variant: str = '_8subj') -> NeuroidAssembly:
    """Load volumetric assembly from local netCDF + stimulus metadata.

    :param split: 'train' or 'test'
    :param variant: '_8subj' (8 subjects, 515 images) or '_4subj' (4 subjects, ~1000 images)
    """
    nc_path = _LOCAL_DIR / f"Allen2022_fmri_{split}{variant}.nc"
    stim_csv = _LOCAL_DIR / f"stimulus_metadata_{split}{variant}.csv"

    meta = pd.read_csv(stim_csv)
    stimuli = StimulusSet(meta)

    stim_paths = {}
    for _, row in meta.iterrows():
        for d in [_LOCAL_DIR / "stimuli_train", _LOCAL_DIR / "stimuli_test"]:
            p = d / row["image_file_name"]
            if p.exists():
                stim_paths[row["stimulus_id"]] = str(p)
                break
    stimuli.stimulus_paths = stim_paths
    stimuli.identifier = f"Allen2022_fMRI_{split}{variant}_Stimuli"
    stimuli.name = stimuli.identifier

    loader = StimulusMergeAssemblyLoader(
        cls=NeuroidAssembly,
        file_path=str(nc_path),
        stimulus_set_identifier=stimuli.identifier,
        stimulus_set=stimuli,
    )
    assembly = loader.load()
    assembly.attrs["identifier"] = f"Allen2022_fMRI_{split}{variant}_Assembly"
    return assembly


# Default (8-subject, 515 images)
data_registry['Allen2022_fmri_train'] = lambda: _load_local_assembly('train', '_8subj')
data_registry['Allen2022_fmri_test'] = lambda: _load_local_assembly('test', '_8subj')

# 4-subject variant (subjects 1,2,5,7; ~1000 images)
data_registry['Allen2022_fmri_4subj_train'] = lambda: _load_local_assembly('train', '_4subj')
data_registry['Allen2022_fmri_4subj_test'] = lambda: _load_local_assembly('test', '_4subj')
