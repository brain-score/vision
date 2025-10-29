from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition  
from brainscore_vision.utils import LazyLoad


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
VISUAL_DEGREES = 8

def _Gifford2022(region, similarity_metric, identifier_metric_suffix):
	number_of_trials = 1
	visual_degrees = VISUAL_DEGREES
	train_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='train', average_repetitions=True))
	test_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=True))
	return TrainTestNeuralBenchmark(identifier=f'Gifford2022.{region}-{identifier_metric_suffix}',
	                          version=1,
	                          ceiling_func=lambda: None,
	                          train_assembly=train_assembly,
	                          test_assembly=test_assembly,
	                          similarity_metric=similarity_metric,
	                          visual_degrees=visual_degrees,
	                          number_of_trials=number_of_trials,
	                          parent=region,
							  bibtex=BIBTEX)    

def Gifford2022(region, metric_type):
    similarity_metric = load_metric(f'{metric_type}-split')
    return _Gifford2022(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type)

def load_assembly(region, split, average_repetitions=True):
	assembly = load_dataset(f'Gifford2022_{split}')
	#TODO don't need to select by region since all channels are combined
	# for now, set the region coordinate to match the expected structure
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0) #TODO: check if intended
	if average_repetitions:
		assembly = average_repetition(assembly)
	#TODO assert VISUAL_DEGREES == assembly.attrs['image_size_degree']
	return assembly
