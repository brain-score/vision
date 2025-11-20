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
    csv_sha1="1a10e75ef9dc9eed6a4eca8e183b81f0d642dda8",
    zip_sha1="7dcc25442ab6737302f96a7b3bd2526afd331637",
    csv_version_id=".6lLxENqxfIICv8uUoG8bqVpk8.SISaW",
    zip_version_id="EVywR8.jXAoABgiyAOBlXhCikvUYNX_.",
    filename_prefix="stimulus_")

stimulus_set_registry['Gifford2022_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_EEG2_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Gifford2022",
    csv_sha1="d7325a55602239e67892b8c596d37e2ee609a59a",
    zip_sha1="bcd0b909df89ff39836ad0bcc9581a79189af2ac",
    csv_version_id="svrZiuFCrOUwlA13h9D78b7CxLJ5MN7q",
    zip_version_id="jhWwXJI.WstIO5mImv7igtGozeYbFFqk",
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