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
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="eedb49ad7c472a2c9fd8e1e71e3c8b44d85ee336",
    zip_sha1="7d01cdb3c91d81d588a07ea408239f9519072ef8",
    csv_version_id="3w83Hzg2xHTPpLioD_CJkZBV5vAK56Xq",
    zip_version_id="txB3V0pTukiL38syytWs38CSxgkrV04x",
    filename_prefix="stimulus_")

stimulus_set_registry['Papale2025_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_TVSD_test_Stimuli",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="f815b363b80cfe0a377918892fa2a001c6fc293f",
    zip_sha1="41cd5a05fe9d5118b05fc6b35654e36ff021570d",
    csv_version_id="1s.vEc8ROnVCsQHUVMimzxeA7XXb.3hF",
    zip_version_id="DfMecEumrHzcRVjc9ssMkswS6LEBRdBh",
    filename_prefix="stimulus_")

data_registry['Papale2025_train'] = lambda: load_assembly_from_s3(
    identifier="THINGS_TVSD_train_Assembly",
    version_id="6Zm3hPGgleb8A0vVOqgopkVRw1PTJEDk",
    sha1="efd8c94616e70e92f8cdc47f50f5dd71da643acc",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_train'))

data_registry['Papale2025_test'] = lambda: load_assembly_from_s3(
    identifier="THINGS_TVSD_test_Assembly",
    version_id="bTBmSYMWp1xMxqAxaE11ThpHb.nvEZFf",
    sha1="cffa371213fff01b78dd841e3582551346c43784",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_test'))
