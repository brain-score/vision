from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmark_helpers.neural_common import timebins_from_assembly  
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel
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

class _Papale2025(BenchmarkBase):
	def __init__(self, region, similarity_metric):
		self._region = region
		self._number_of_trials = 1
		self._visual_degrees = VISUAL_DEGREES
		self.train_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='train'))
		self.test_assembly = LazyLoad(lambda region=region: load_assembly(region=region, split='test'))
		self._similarity_metric = similarity_metric
		super().__init__(
			identifier=f'Papale2025.{region}-pls', 
			version=1,
			ceiling_func=lambda: 1,  # TODO: replace with actual ceiling
			parent=region,
			bibtex=BIBTEX
		)
	def __call__(self, candidate : BrainModel):

		# get the activations from the train set
		train_stimulus_set = self.train_assembly.stimulus_set
		timebins = timebins_from_assembly(self.train_assembly)
		candidate.start_recording(self._region, time_bins=timebins)
		stimulus_set = place_on_screen(train_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                    source_visual_degrees=self._visual_degrees)
		train_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
		
		# get the activations from the test set
		test_stimulus_set = self.test_assembly.stimulus_set
		timebins = timebins_from_assembly(self.test_assembly)
		candidate.start_recording(self._region, time_bins=timebins)
		stimulus_set = place_on_screen(test_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
									source_visual_degrees=self._visual_degrees)
		test_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

		raw_score = self._similarity_metric(source_train=train_activations, source_test=test_activations,
                target_train=self.train_assembly, target_test=self.test_assembly)
		#TODO ceil score
		return raw_score
        

def Papale2025PLS(region):
    similarity_metric = load_metric('pls-split')
    return _Papale2025(region, similarity_metric=similarity_metric)

def Papale2025Ridge(region):
	similarity_metric = load_metric('ridge-split')
	return _Papale2025(region, similarity_metric=similarity_metric)

def Papale2025NeuronToNeuron(region):
	similarity_metric = load_metric('neuron_to_neuron-split')
	return _Papale2025(region, similarity_metric=similarity_metric)

def load_assembly(region, split):
	assembly = load_dataset(f'Papale2025_{split}')
	assembly = assembly.sel(region=region)
	assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
	assembly.load()
	assembly = assembly.isel(time_bin=0) #TODO: check if intended
	#TODO assert VISUAL_DEGREES == assembly.attrs['image_size_degree']
	return assembly
