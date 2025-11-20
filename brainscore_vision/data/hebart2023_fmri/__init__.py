from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

BIBTEX = """@article{hebart_things-data_2023,
	title = {{THINGS}-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
	volume = {12},
	issn = {2050084X},
	doi = {10.7554/eLife.82580},
	journal = {eLife},
	author = {Hebart, M. N. and Contier, O. and Teichmann, L. and Rockter, A. H. and Zheng, C. Y. and Kidder, A. and Corriveau, A. and Vaziri-Pashkam, M. and Baker, C. I.},
	year = {2023},
	pmid = {36847339},
}"""

stimulus_set_registry['Hebart2023_fmri_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_fMRI_train_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Hebart2023_fmri",
    csv_sha1="b424b1a55595a4666fbc140a5a801fcd184d1a44",
    zip_sha1="1c65c28c104e100e6e3fe2128656abe647e41bd9",
    csv_version_id="gioa1wYVzJVhnAOPJbj4q9ZdoCeWnUYX",
    zip_version_id="nRsfLuqfOMAoQ5qGWbKxKKW_WGtgGJGS",
    filename_prefix="stimulus_")

stimulus_set_registry['Hebart2023_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_fMRI_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Hebart2023_fmri",
    csv_sha1="0ef71c62210d0a0bf91cb2cd8f0e1404477e0e3a",
    zip_sha1="41cd5a05fe9d5118b05fc6b35654e36ff021570d",
    csv_version_id="U6krcZSjVvO8sjPsTsGAaiOxw4Ozai1d",
    zip_version_id="11dAYDGOdwzma61sIZ.u3weJ.x7qcwaN",
    filename_prefix="stimulus_")

data_registry['Hebart2023_fmri_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_fMRI_train_Assembly",
    version_id="U.I.Wsn52bZKdNQ9Y_OdisvCfkgMG7Ab",
    sha1="6d249e7cf4804ff78e9a38e482148cad74f90d19",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Hebart2023_fmri",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_train'))

data_registry['Hebart2023_fmri_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_fMRI_test_Assembly",
    version_id="LksBmANKvDyfqkl0U5Dy33Q4VAHOO5g3",
    sha1="948bd3104fc8f0f0a15180f030666c74d69de2b3",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Hebart2023_fmri",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_test'))