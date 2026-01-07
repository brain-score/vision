from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import TrainTestNeuralBenchmark, average_repetition, flatten_timebins_into_neuroids
from brainscore_vision.utils import LazyLoad
import numpy as np

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

### Note on VISUAL_DEGREES
# the human subjects were shown 500x500 pixel images
# "each of the 20 images was presented centrally with a visual angle of 7° for 100 ms" (Gifford et al, 2022 p.2)
VISUAL_DEGREES = 7

### NOTE on ridge regression alphas
# You can find the default list of alphas at:
# brainscore_vision.metrics.regression_correlation.metric.ALPHA_LIST
# Due to having many neuroids (100 time points x 17 electrodes flattened into neuroids),
# and many subjects (10), we use a reduced list of alphas here to speed up computation:
REDUCED_ALPHA_LIST = [
    *[0.01, 0.1, 0.5, 1.0, 10, 50, 100, 500, 1000, 5000],
    *np.linspace(1e4, 1e5, 4, endpoint=False),
    *np.linspace(1e5, 1e6, 4, endpoint=False),
    *np.linspace(1e6, 1e7, 5)
]


def _Gifford2022(region, 
				similarity_metric, 
				identifier_metric_suffix,
				alpha_coord=None,
				per_voxel_ceilings=False,
				visual_degrees=VISUAL_DEGREES,
				ceiler = load_metric('internal_consistency')):
	number_of_trials = 1
	train_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='train', average_repetitions=True))
	test_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=True))
	test_assembly_repetition = LazyLoad(lambda region=region: load_assembly(region=region, split='test', average_repetitions=False))
	return TrainTestNeuralBenchmark(identifier=f'Gifford2022.{region}-{identifier_metric_suffix}',
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

def Gifford2022(region, metric_type, alphas=REDUCED_ALPHA_LIST):
    similarity_metric = load_metric(f'{metric_type}_split', alphas=alphas)
    return _Gifford2022(region, similarity_metric=similarity_metric, identifier_metric_suffix=metric_type,
					   alpha_coord='subject', per_voxel_ceilings=False)

def load_assembly(region, split, average_repetitions=True):
	assembly = load_dataset(f'Gifford2022_{split}')
	#NOTE: brainscore only supports regions 'V1', 'V2', 'V4', 'IT', so EEG data is mapped to IT for now
	assembly.load()
	
	# predict for each electrode, individual time bins in a window of 0s to 0.6s relative to stimulus onset
	assembly = assembly.sel(time_bin=slice(0, 0.59))
	assert assembly.shape[2] == 60, f"selected {assembly.shape[2]} time bins, expected 60"
	
	# flatten the time points into the neuroid dimension
	assembly = flatten_timebins_into_neuroids(assembly)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly = assembly.isel(time_bin=0)  # remove time dimension

	if average_repetitions:
		assembly = average_repetition(assembly)
	return assembly
