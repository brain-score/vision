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
    identifier="THINGS_EEG2_train_Stimuli",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="310758f340c1bc94661c586f37418b261505607a",
    zip_sha1="7dcc25442ab6737302f96a7b3bd2526afd331637",
    csv_version_id="_l1oIWPJRbD8_rp6fgNcP2vTpDimgOzr",
    zip_version_id="MNRak8VOx8f97sARnE7L.woH4qhMs38d",
    filename_prefix="stimulus_")

stimulus_set_registry['Gifford2022_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_EEG2_test_Stimuli",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="c094407c2a8d921e127c8788a490f500f51e86af",
    zip_sha1="bcd0b909df89ff39836ad0bcc9581a79189af2ac",
    csv_version_id="D_JHZhWXkG4rZADCB3f0cd12DWEpkBg0",
    zip_version_id="yTM_Ir03BlcvoF5yAnR73PkKtDYY3BGu",
    filename_prefix="stimulus_")

data_registry['Gifford2022_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_EEG2_train_Assembly",
    version_id="3gh81MGgx6DjvENpbRgHovj_5Mis4YJM",
    sha1="53748acfb1e15cf39a2db54edc4dfa91d1736f31",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_train'))

data_registry['Gifford2022_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_EEG2_test_Assembly",
    version_id="VYvXUxWz.nfRUPonJMJ_ifEgFht5eXRK",
    sha1="307d1222abb39d3ca8a36d06b8b9769bacef130d",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_test'))