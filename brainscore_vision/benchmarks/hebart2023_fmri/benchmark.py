from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition, filter_reliable_neuroids
from brainscore_vision.metrics.regression_correlation.metric import ALPHA_LIST
from brainscore_vision.utils import LazyLoad


BIBTEX = """@article{hebart_things-data_2023,
	title = {{THINGS}-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
	volume = {12},
	issn = {2050084X},
	doi = {10.7554/eLife.82580},
	journal = {eLife},
	author = {Hebart, M. N. and Contier, O. and Teichmann, L. and Rockter, A. H. and Zheng, C. Y. and Kidder, A. and Corriveau, A. and Vaziri-Pashkam, M. and Baker, C. I.},
	year = {2023},
	pmid = {36847339},
}"""

### Note on VISUAL_DEGREES
# all images were shown at 10 visual degrees, with a fixation cross spanning 0.5 visual degrees in the center 
VISUAL_DEGREES = 10


NOISE_CEILING_THRESHOLD = 0.3 * 100

def _Hebart2023fmri(region,
					similarity_metric,
					identifier_metric_suffix,
					alpha_coord=None,
					per_voxel_ceilings=False,
					visual_degrees=VISUAL_DEGREES,
					ceiler = load_metric('internal_consistency'),
					noise_ceiling_threshold=NOISE_CEILING_THRESHOLD):
	number_of_trials = 1
	train_assembly = LazyLoad(lambda region=region, nct=noise_ceiling_threshold: 
						   		load_assembly(region=region, 
											split='train', 
											average_repetitions=False,  # train has no repetitions
											noise_ceiling_threshold=nct))  
	test_assembly = LazyLoad(lambda region=region, nct=noise_ceiling_threshold: 
						  		load_assembly(region=region, 
											split='test', 
											average_repetitions=True, 
											noise_ceiling_threshold=nct))
	test_assembly_repetition = LazyLoad(lambda region=region, nct=noise_ceiling_threshold: 
								load_assembly(region=region, 
											split='test', 
											average_repetitions=False, 
											noise_ceiling_threshold=nct))
	return TrainTestNeuralBenchmark(identifier=f'Hebart2023_fmri.{region}-{identifier_metric_suffix}',
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

def Hebart2023fmri(region, metric_type, alphas=ALPHA_LIST):
    similarity_metric = load_metric(f'{metric_type}_split', alphas=alphas)
    return _Hebart2023fmri(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type,
						   alpha_coord='subject', per_voxel_ceilings=False)

def load_assembly(region, split, average_repetitions, noise_ceiling_threshold=NOISE_CEILING_THRESHOLD):
	assembly = load_dataset(f'Hebart2023_fmri_{split}')
	assembly = filter_reliable_neuroids(assembly, noise_ceiling_threshold, 'nc_testset')
	assembly = assembly.sel(region=region)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0)
	if average_repetitions:
		assembly = average_repetition(assembly)
	return assembly
