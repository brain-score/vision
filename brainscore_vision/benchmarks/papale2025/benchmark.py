from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition, filter_reliable_neuroids
from brainscore_vision.metrics.regression_correlation.metric import ALPHA_LIST
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
### Note on VISUAL_DEGREES
# Papale2025 showed 500x500 pixel images on 1024x768 pixel monitors for 200ms
# They used different monitors and different viewing distances for the two monkeys:
# monkey N: viewing distance 47cm, 28.9 pixels per degree => 17.3 visual degrees
# monkey F: viewing distance 58cm, 26.9 pixels per degree => 18.6 visual degrees

# Es a compromise between the two subjects, we set visual degrees to 18
# Beware that for models, that have a small field of view (e.g. the default 8 visual degrees),
# a large part of the image will be cropped away!
VISUAL_DEGREES = 18


RELIABILITY_THRESHOLD = 0.3

def _Papale2025(region, 
				similarity_metric, 
				identifier_metric_suffix,
				alpha_coord=None, 
				per_voxel_ceilings=False,
				visual_degrees=VISUAL_DEGREES,
				ceiler = load_metric('internal_consistency'),
				reliability_threshold=RELIABILITY_THRESHOLD):
	number_of_trials = 1
	train_assembly = LazyLoad(lambda region=region, rt=reliability_threshold: 
						   		load_assembly(region=region, 
											split='train', 
											average_repetitions=False,  # train has no repetitions
											reliability_threshold=rt))  
	test_assembly = LazyLoad(lambda region=region, rt=reliability_threshold: 
						  		load_assembly(region=region, 
											split='test', 
											average_repetitions=True, 
											reliability_threshold=rt))
	test_assembly_repetition = LazyLoad(lambda region=region, rt=reliability_threshold: 
								load_assembly(region=region, 
					  						split='test', 
											average_repetitions=False, 
											reliability_threshold=rt))
	return TrainTestNeuralBenchmark(identifier=f'Papale2025.{region}-{identifier_metric_suffix}',
	                          version=2,
	                          ceiling_func=lambda: ceiler(test_assembly_repetition),
	                          train_assembly=train_assembly,
	                          test_assembly=test_assembly,
	                          similarity_metric=similarity_metric,
							  alpha_coord=alpha_coord,
							  per_voxel_ceilings=per_voxel_ceilings,
	                          visual_degrees=visual_degrees,
	                          number_of_trials=number_of_trials,
	                          parent=region,
							  bibtex=BIBTEX)    

def Papale2025(region, metric_type, alphas=ALPHA_LIST):
    similarity_metric = load_metric(f'{metric_type}_split', alphas=alphas)
    return _Papale2025(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type,
					   alpha_coord='subject', per_voxel_ceilings=False)


def load_assembly(region, split, average_repetitions, reliability_threshold=RELIABILITY_THRESHOLD):
	assembly = load_dataset(f'Papale2025_{split}')
	assembly = filter_reliable_neuroids(assembly, reliability_threshold, 'reliability')
	assembly = assembly.sel(region=region)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0)
	if average_repetitions:
		assembly = average_repetition(assembly)
	return assembly
