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

_BUCKET = "brainscore-storage/brainscore-vision/benchmarks/Allen2022/Allen2022_fmri"

# -- Stimulus sets: 8-subject (515 images: 412 train + 103 test) ---------------

stimulus_set_registry['Allen2022_fmri_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_train_Stimuli",
    bucket=_BUCKET,
    csv_sha1="b9119b9c430877ad2e0786888e0bb66fa50529c7",
    zip_sha1="158c4813355b10fa05130b070b7c45056d7cd5d6",
    csv_version_id="EoSt2r7Gzpj_K84RT8hVCbl0zoP3I3FK",
    zip_version_id="sasKQwanv3iv8GgoM10JkwqbD1UKMFpj",
    filename_prefix="stimulus_")

stimulus_set_registry['Allen2022_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_test_Stimuli",
    bucket=_BUCKET,
    csv_sha1="cfb0ae15fd841f660d4f9d77f2a13213a01f2078",
    zip_sha1="980f66d404f0bfbc898ffd8b299e9f9dd7211a9e",
    csv_version_id="FW75Q.eWtrETXk8B006W85G653J52F1K",
    zip_version_id="km7bSBf3iKc2F2p4BD4OB4z9UORIGQRf",
    filename_prefix="stimulus_")

# -- Stimulus sets: 4-subject (1000 images: 800 train + 200 test) --------------

stimulus_set_registry['Allen2022_fmri_4subj_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_4subj_train_Stimuli",
    bucket=_BUCKET,
    csv_sha1="c4037d0c6ba9bf69ecfb2571bee8d3b494b12d14",
    zip_sha1="c1547e3747c4963a77db23184a70a195d2ccb92d",
    csv_version_id="WoseNY.dEQJ4u7GEWPQGgCsqyCDOLJqg",
    zip_version_id="cWHAMDTXkmhGH0NdaVldk437p16fbrTg",
    filename_prefix="stimulus_")

stimulus_set_registry['Allen2022_fmri_4subj_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="Allen2022_fMRI_4subj_test_Stimuli",
    bucket=_BUCKET,
    csv_sha1="4259c8ddffed135dc73d0968aaeff66046355250",
    zip_sha1="c489db0d7f0ec823ca28ecc53a7c25123fe851dd",
    csv_version_id="ElpofOBBOnY_RAD6yIxwmbOMQiQTgcV_",
    zip_version_id="W00CwAFvSofEdQSAkGtTZcCMQt8oLo7G",
    filename_prefix="stimulus_")

# -- Assemblies: 8-subject (default, 515 images) -------------------------------

data_registry['Allen2022_fmri_train'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_train_Assembly",
    version_id="ZAHUcNkJV4zvRCJfrXbjqmVM9OgyQVE2",
    sha1="5919ec2838f69bfb5d87e16f197868550713cde0",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_train'))

data_registry['Allen2022_fmri_test'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_test_Assembly",
    version_id="27mm9BVBp_nyq6zlrywUtcR_TxePQCid",
    sha1="3203ab79d48fab9b6e31f9d2a6301a11c512a0e9",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_test'))

# -- Assemblies: 4-subject (subjects 1,2,5,7; ~1000 images) --------------------

data_registry['Allen2022_fmri_4subj_train'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_4subj_train_Assembly",
    version_id="Dn4FwRGPUCmRmvt6ybn501DkLU.PmT42",
    sha1="db4493972a0fcc36757b9decda1173e9b9ab7a09",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_4subj_stim_train'))

data_registry['Allen2022_fmri_4subj_test'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_4subj_test_Assembly",
    version_id="opZA1Oyvx9sQWTCfQdC9djv7Be0p8WzJ",
    sha1="8b1a5e6378eff6a7499f6cb1ab0e3806d567bb4e",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_4subj_stim_test'))
