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
    bucket="brainscore-storage/brainscore-vision/benchmarks/Gifford2022",
    csv_sha1="a406afab99abca28b141bd4457daea8f3b001f75",
    zip_sha1="a46d5476b3caef89b481bdf50c70ecd9cac28fa4",
    csv_version_id="erlnWZh_o9Mk17tTx3id0Kq_G.2k4CRY",
    zip_version_id="T70I8pGBqcxt4bpdYfm0KU_iD67_3ryu",
    filename_prefix="stimulus_")

stimulus_set_registry['Gifford2022_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_EEG2_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Gifford2022",
    csv_sha1="f79dd731e36a9e5c1b62fc6518ff30c8d71df182",
    zip_sha1="a1cdd468f02309dd805466219c019f1a7a2ffb44",
    csv_version_id="QCHnkzt7S.8ik.kuMORgxTT24Y4QA3zs",
    zip_version_id="4rTsFRxCmXcq7hAo.CCQgGoFXMWOQoVB",
    filename_prefix="stimulus_")

data_registry['Gifford2022_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_EEG2_train_Assembly",
    version_id=".hWjA5kCrxTbX.0b4s56ABGtk5aVbBm.",
    sha1="412002748b9fd4e61a279b4eb481964e9b309b5e",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Gifford2022",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_train'))

data_registry['Gifford2022_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_EEG2_test_Assembly",
    version_id="2kux2JDyRBvJD4Y3mjBW4hqytl8FX5SJ",
    sha1="8103ade9df6161566d885e9265ee62e0130b588b",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Gifford2022",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Gifford2022_stim_test'))