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

# Local path for pre-S3-upload development.
# Replace with load_assembly_from_s3 + proper SHA1 hashes after upload.
_LOCAL_DIR = Path("/Volumes/Hagibis/nsd/brainscore_surface")

# Reuse the same stimulus sets as volumetric (identical COCO images).
# Stimulus sets are registered by the volumetric data package (allen2022_fmri).
# If running standalone, they need to be registered here too.
if 'Allen2022_fmri_stim_train' not in stimulus_set_registry:
    stimulus_set_registry['Allen2022_fmri_stim_train'] = lambda: load_stimulus_set_from_s3(
        identifier="Allen2022_fMRI_train_Stimuli",
        bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
        csv_sha1="b30e2ed9730c905b9b8673155230b762571e71d5",
        zip_sha1="8aba5aefed91ea153451ddcba79c08eece627ed2",
        csv_version_id="null",
        zip_version_id="null",
        filename_prefix="stimulus_")

if 'Allen2022_fmri_stim_test' not in stimulus_set_registry:
    stimulus_set_registry['Allen2022_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
        identifier="Allen2022_fMRI_test_Stimuli",
        bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
        csv_sha1="d5d39fa5fa492c0249b075bdc7e2a514b239accf",
        zip_sha1="78276b6a4f32261f211ce4d3fa51294e8f14d30b",
        csv_version_id="null",
        zip_version_id="null",
        filename_prefix="stimulus_")


def _load_local_assembly(split: str) -> NeuroidAssembly:
    """Load surface assembly from local netCDF + stimulus metadata.

    Used during development before S3 upload. After upload, replace the
    data_registry entries below with load_assembly_from_s3 calls.
    """
    nc_path = _LOCAL_DIR / f"Allen2022_fmri_surface_{split}.nc"
    stim_csv = _LOCAL_DIR / f"stimulus_metadata_{split}.csv"
    stim_dir = _LOCAL_DIR / f"stimuli_{split}"

    # Build StimulusSet from local files
    meta = pd.read_csv(stim_csv)
    stimuli = StimulusSet(meta)
    stimuli.stimulus_paths = {
        row["stimulus_id"]: str(stim_dir / row["image_file_name"])
        for _, row in meta.iterrows()
    }
    stimuli.identifier = f"Allen2022_fMRI_surface_{split}_Stimuli"
    stimuli.name = stimuli.identifier

    # Load assembly through the standard brainio loader chain
    loader = StimulusMergeAssemblyLoader(
        cls=NeuroidAssembly,
        file_path=str(nc_path),
        stimulus_set_identifier=stimuli.identifier,
        stimulus_set=stimuli,
    )
    assembly = loader.load()
    assembly.attrs["identifier"] = f"Allen2022_fMRI_surface_{split}_Assembly"
    return assembly


# Surface neural assemblies -- local loading (pre-S3 upload)
data_registry['Allen2022_fmri_surface_train'] = lambda: _load_local_assembly('train')
data_registry['Allen2022_fmri_surface_test'] = lambda: _load_local_assembly('test')
