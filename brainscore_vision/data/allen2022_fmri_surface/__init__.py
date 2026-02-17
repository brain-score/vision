from brainscore_vision import data_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3
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

_BUCKET = "brainscore-storage/brainscore-vision/benchmarks/Allen2022/Allen2022_fmri_surface"

# Surface assemblies reuse the volumetric stimulus sets (identical COCO images).
# Stimulus set registry keys are defined in brainscore_vision.data.allen2022_fmri.

# -- Assemblies: 8-subject (default, 515 images) -------------------------------

data_registry['Allen2022_fmri_surface_train'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_surface_train_Assembly",
    version_id="Ug1qir_TigXWwBuoamnDHj1r1Ca5FlEM",
    sha1="67241ed5a84d8e7ae77f8c8d72e7219508fb9d7e",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_train'))

data_registry['Allen2022_fmri_surface_test'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_surface_test_Assembly",
    version_id="XwzoT6lBFvUnn9l6WnD9GPPw_PafqAQ9",
    sha1="8b4fcc132b5c1e81bb61e38ca51875a98619066b",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_stim_test'))

# -- Assemblies: 4-subject (subjects 1,2,5,7; ~1000 images) --------------------

data_registry['Allen2022_fmri_surface_4subj_train'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_surface_4subj_train_Assembly",
    version_id="SiHRp_P1RPH6BB4XJbRz4GPiYD4jxIJJ",
    sha1="5c1e7ab2d1444c140028599f8a30aeae34297c87",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_4subj_stim_train'))

data_registry['Allen2022_fmri_surface_4subj_test'] = lambda: load_assembly_from_s3(
    identifier="Allen2022_fMRI_surface_4subj_test_Assembly",
    version_id="kzuIe9w5y7AfuajgzUD1SYosbBs0tC69",
    sha1="1f8524de28ad6cd3a89feddcfb82d9ab67dd9656",
    bucket=_BUCKET,
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Allen2022_fmri_4subj_stim_test'))
