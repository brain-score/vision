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
    csv_sha1="4a094a7e4c2408abc333ba610215a7c30277d904",
    zip_sha1="0a97a7952097800207a23fe73281c256dfc90902",
    csv_version_id="kCt7PZx9jxXDrRhyow9y8hDUUDFAAp._",
    zip_version_id="S.DCjiAaGQ6PjCUs4QugwsHaZa9YiEuV",
    filename_prefix="stimulus_")

stimulus_set_registry['Papale2025_stim_test'] = lambda: load_stimulus_set_from_s3(
    identifier="THINGS_TVSD_test_Stimuli",
    bucket="brainscore-storage/brainscore-vision/benchmarks/Papale2025",
    csv_sha1="b08c1c560f008cf3fb2982c22c881ea8ffce0410",
    zip_sha1="dfd6e8769414505afbf51de3ab207cb7a3cf49d6",
    csv_version_id="gEGUgT7PJB8z3roRkTiJ6zriSuehd2DI",
    zip_version_id="AGNDfuS5nyIjTuXrM4hWFze.sWM9py2Z",
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
