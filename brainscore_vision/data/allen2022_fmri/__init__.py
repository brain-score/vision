from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import (
    load_stimulus_set_from_s3, load_assembly_from_s3,
)
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

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

# Neural assemblies
data_registry['Allen2022_fmri_train'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_train_Assembly",
    version_id="null",
    sha1="5fd6b9383ace89842c17f413d00591730a0e0216",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_train'))

data_registry['Allen2022_fmri_test'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_test_Assembly",
    version_id="null",
    sha1="3e4b0c056d05caab4b9146ec3262cd6a2361b6b1",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Allen2022_fmri",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_test'))
