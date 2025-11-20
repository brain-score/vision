from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

BIBTEX = """@article{papale_extensive_2025,
	title = {An extensive dataset of spiking activity to reveal the syntax of the ventral stream},
	volume = {113},
	issn = {08966273},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S089662732400881X},
	doi = {10.1016/j.neuron.2024.12.003},
	journal = {Neuron},
	author = {Papale, Paolo and Wang, Feng and Self, Matthew W. and Roelfsema, Pieter R.},
	year = {2025},
}"""

stimulus_set_registry['Papale2025_stim_train'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_TVSD_train_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Papale2025",
    csv_sha1="955b18a10b887550b91b7a03eacc3fc413ccb05e",
    zip_sha1="7d01cdb3c91d81d588a07ea408239f9519072ef8",
    csv_version_id=".AiRrc3NRAAPoEDAK1zY5LiU_H2p9CQr",
    zip_version_id="mSEP5DgYmtu_FneAinxPS3gB0cS9bC1M",
    filename_prefix="stimulus_")

stimulus_set_registry['Papale2025_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_TVSD_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Papale2025",
    csv_sha1="3d5fefd685b3b8684f77856da6848be10473d677",
    zip_sha1="41cd5a05fe9d5118b05fc6b35654e36ff021570d",
    csv_version_id="c1XCWg79sI2vVevdqp590NbQJxqyKaSb",
    zip_version_id="4ff5Ire9_Q.3MLXUgBnA7UJoweQHEyDG",
    filename_prefix="stimulus_")

data_registry['Papale2025_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_TVSD_train_Assembly",
    version_id="q3sE0RZ5Fuq0C6vTVIiF48AsvG40.b7l",
    sha1="97e3e9ed2280441a30fa5c33e8537db1b45a9da8",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Papale2025",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_train'))

data_registry['Papale2025_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_TVSD_test_Assembly",
    version_id="XK0CVCAzfHNOrKgmNrLqFfwvs_gSWKb.",
    sha1="685025f67e90006e775ac153e8e0007e35b62bbc",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Papale2025",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_test'))
