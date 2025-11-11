from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition
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
VISUAL_DEGREES = 8

def _Hebart2023fmri(region, similarity_metric, identifier_metric_suffix):
	number_of_trials = 1
	visual_degrees = VISUAL_DEGREES
	train_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='train', average_repetitions=False))  # train has no repetitions
	test_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=True))
	test_assembly_repetition = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=False))
	ceiler = load_metric('internal_consistency')
	return TrainTestNeuralBenchmark(identifier=f'Hebart2023_fmri.{region}-{identifier_metric_suffix}',
	                          version=1,
	                          ceiling_func=lambda: ceiler(test_assembly_repetition),
	                          train_assembly=train_assembly,
	                          test_assembly=test_assembly,
	                          similarity_metric=similarity_metric,
	                          visual_degrees=visual_degrees,
	                          number_of_trials=number_of_trials,
	                          parent=region,
							  bibtex=BIBTEX)    

def Hebart2023fmri(region, metric_type):
    similarity_metric = load_metric(f'{metric_type}_split')
    return _Hebart2023fmri(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type)

def load_assembly(region, split, average_repetitions):
	assembly = load_dataset(f'Hebart2023_fmri_{split}')
	assembly = assembly.sel(region=region)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0)
	if average_repetitions:
		assembly = average_repetition(assembly)
	return assembly
