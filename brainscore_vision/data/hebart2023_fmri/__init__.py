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
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="50d29cdc1e32a745613ce01bc2e62b5bd622778e",
    zip_sha1="1c65c28c104e100e6e3fe2128656abe647e41bd9",
    csv_version_id="ufJOv.gfNlmnaA6V.4NEV5nawj4Jengi",
    zip_version_id="Zuj85ERewjMHsSQu9JyDqm7er2dzSxMb",
    filename_prefix="stimulus_")

stimulus_set_registry['Hebart2023_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_fMRI_test_Stimuli",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="10e606dc6ddf00aa556d5450330af13191d74505",
    zip_sha1="41cd5a05fe9d5118b05fc6b35654e36ff021570d",
    csv_version_id=".sRmu_hPqnrlEzXA2Isr2hQdQNyWVF8n",
    zip_version_id="_KbhXIG9nV3cdsUXTwctgU68tA9jfqfW",
    filename_prefix="stimulus_")

data_registry['Hebart2023_fmri_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_fMRI_train_Assembly",
    version_id="BNw3JgSbJ88YSU6PuMzTbd854jSOZja_",
    sha1="a9c2d597c0476adf851bcfd73c5f53d069b68742",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_train'))

data_registry['Hebart2023_fmri_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_fMRI_test_Assembly",
    version_id="KJzP6phj01ob5satRFDZ6vWPZOyxWIE0",
    sha1="290d9bfd822b5775039ff736cf312d3488045198",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_test'))