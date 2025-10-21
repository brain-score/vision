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
    identifier='THINGS_fRMRI_train_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='8007cfd5b5e352d5551fb24616d8cf39acfd87ad',
    zip_sha1='1c65c28c104e100e6e3fe2128656abe647e41bd9',
    csv_version_id='h.LqA.NpZR6fxaSVwdii73rmEaVgSE.3',
    zip_version_id='ARQrplSY1PM5NbvhY9fd.hvoXUomn_CY',
    filename_prefix='stimulus_')

stimulus_set_registry['Hebart2023_fmri_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier='THINGS_fRMRI_test_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='7b867fbebd24552ec558bd0a340aac7e326a6b24',
    zip_sha1='41cd5a05fe9d5118b05fc6b35654e36ff021570d',
    csv_version_id='uofTv9osVVOtrB4bYn4nHAmoZs6YOdw9',
    zip_version_id='R_FsZlIP2mAD7VPrvEzAvwze17d95ELI',
    filename_prefix='stimulus_')

data_registry['Hebart2023_fmri_train'] = lambda: load_assembly_from_s3(
    identifier='THINGS_fMRI_train_Assembly',
    version_id='pSyQKjGGeI2n4p5NDK2KMpXwZP2AgaTF',
    sha1='c8a38e2a3f42889b2469b6d23607bd1e899f6981',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_train'))

data_registry['Hebart2023_fmri_test'] = lambda: load_assembly_from_s3(
    identifier='THINGS_fMRI_test_Assembly',
    version_id='aWWelb.6p8GRG8RlvV_0uWGMyCswOQ7t',
    sha1='ed84b471bdf791975f401c3096b5a2274aac9cec',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Hebart2023_fmri_stim_test'))