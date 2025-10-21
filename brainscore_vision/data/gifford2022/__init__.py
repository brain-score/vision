from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

BIBTEX = """@article{gifford_large_2022,
	title = {A large and rich {EEG} dataset for modeling human visual object recognition},
	volume = {264},
	issn = {10538119},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S1053811922008758},
	doi = {10.1016/j.neuroimage.2022.119754},
	journal = {NeuroImage},
	author = {Gifford, Alessandro T. and Dwivedi, Kshitij and Roig, Gemma and Cichy, Radoslaw M.},
	year = {2022},
	}"""

stimulus_set_registry['Gifford2022_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier='THINGS_EEG2_train_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='b116056512e9759801c898d1271bcb5872fd3e6e',
    zip_sha1='7dcc25442ab6737302f96a7b3bd2526afd331637',
    csv_version_id='BGM3XoelUJYhtXq7mT4izEVaNCnCd9sA',
    zip_version_id='8DuLsDVWw83ZlgYY_CV0PPYSBAjoocin',
    filename_prefix='stimulus_')

stimulus_set_registry['Gifford2022_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier='THINGS_EEG2_test_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='7f836077c39cbe5733eaaac0bae43e0426cee043',
    zip_sha1='bcd0b909df89ff39836ad0bcc9581a79189af2ac',
    csv_version_id='NJdJ9y9MaucNlnv0JSdRhO34OU5LmiDV',
    zip_version_id='xUPH.BGTrPRrN0EbhpdKO.zMu7k3Kss3',
    filename_prefix='stimulus_')

data_registry['Gifford2022_train'] = lambda: load_assembly_from_s3(
    identifier='THINGS_EEG2_train_Assembly',
    version_id='pcdWzZ9HZm0DYr26Nn3QxSDQ9.AFh3zg',
    sha1='7d13287871a600ca1124ed1ec423b08b147b68a3',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_train'))

data_registry['Gifford2022_test'] = lambda: load_assembly_from_s3(
    identifier='THINGS_EEG2_test_Assembly',
    version_id='AG6ALmlsEEzdBrfwfm5UwL6Es2uSO_0P',
    sha1='91e7d1b229a1eff0299d22f90907940093d6c5e7',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_test'))