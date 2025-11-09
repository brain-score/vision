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
    identifier='THINGS_TVSD_train_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='2aa534d1f80274c384e2aedbb3c73e398f29db5e',
    zip_sha1='7d01cdb3c91d81d588a07ea408239f9519072ef8',
    csv_version_id='J51ZKRuKXtVjRR8YS2cn2Ft1YIwAnH7Z',
    zip_version_id='hhSfdK559QMzegyeo_vRNT0X5kL8P4WY',
    filename_prefix='stimulus_')

stimulus_set_registry['Papale2025_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier='THINGS_TVSD_test_Stimuli',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1='f2b69b2a3f6855b979824617b92932f8acf38f61',
    zip_sha1='41cd5a05fe9d5118b05fc6b35654e36ff021570d',
    csv_version_id='gPwS.nDf7z67SdgIKBUqPjFCEykJULOg',
    zip_version_id='TGPsmZHGcVpogRK1RdPbN.Sw49UubOiZ',
    filename_prefix='stimulus_')

data_registry['Papale2025_train'] = lambda: load_assembly_from_s3(
    identifier='THINGS_TVSD_train_Assembly',
    version_id='HymiZdl2zccTe4alNfpCs1sKAyifXafi',
    sha1='848d61cf0c2b3eb0c873f07f6295e59366b972c5',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_train'))

data_registry['Papale2025_test'] = lambda: load_assembly_from_s3(
    identifier='THINGS_TVSD_test_Assembly',
    version_id='L76NrixHL1lVqLMiyrVmIaulRr.rHixe',
    sha1='1821f4685537eba0484a26f569c204a9e81a641b',
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Papale2025_stim_test'))
