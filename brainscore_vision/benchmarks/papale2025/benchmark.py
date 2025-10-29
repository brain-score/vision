from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition
from brainscore_vision.utils import LazyLoad



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
VISUAL_DEGREES = 8

def _Papale2025(region, similarity_metric, identifier_metric_suffix):
	number_of_trials = 1
	visual_degrees = VISUAL_DEGREES
	train_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='train', average_repetitions=False)) #train has no repetitions
	test_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=True))
	return TrainTestNeuralBenchmark(identifier=f'Papale2025-{region}-{identifier_metric_suffix}',
	                          version=1,
	                          ceiling_func=lambda: 1.0,
	                          train_assembly=train_assembly,
	                          test_assembly=test_assembly,
	                          similarity_metric=similarity_metric,
	                          visual_degrees=visual_degrees,
	                          number_of_trials=number_of_trials,
	                          parent=region,
							  bibtex=BIBTEX)    

def Papale2025(region, metric_type):
    similarity_metric = load_metric(f'{metric_type}-split')
    return _Papale2025(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type)

def load_assembly(region, split, average_repetitions):
	assembly = load_dataset(f'Papale2025_{split}')
	assembly = assembly.sel(region=region)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0) #TODO: check if intended
	if average_repetitions:
		assembly = average_repetition(assembly)
	#TODO assert VISUAL_DEGREES == assembly.attrs['image_size_degree']
	return assembly
