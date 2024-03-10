import numpy as np
import os
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainscore_vision import load_ceiling, load_metric, load_dataset
from brainscore_vision.benchmark_helpers.neural_common import average_repetition, timebins_from_assembly
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase, ceil_score
from brainscore_vision.metrics.internal_consistency import InternalConsistency
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment, LayerSelection, RegionLayerMap
from brainscore_vision.model_interface import BrainModel


def check_brain_models(module):
    module = __import__(module)
    for model in module.get_model_list():
        model = module.get_model(model)
        assert model is not None
        assert isinstance(model, BrainModel)
        check_brain_model_processing(model)
    print('Test successful, you are ready to submit!')


def check_brain_model_processing(model):
    benchmark = _MockBenchmark()
    score = benchmark(model, do_behavior=True)
    assert score is not None


def check_base_models(module):
    module = __import__(module)
    for model in module.get_model_list():
        layers = module.get_layers(model)
        assert layers is not None
        assert isinstance(layers, list)
        assert len(layers) > 0
        assert module.get_model(model) is not None
        check_processing(model, module)
        print('Test successful, you are ready to submit!')


def check_processing(model_identifier, module):
    os.environ['RESULTCACHING_DISABLE'] = '1'
    model_instance = module.get_model(model_identifier)
    layers = module.get_layers(model_identifier)
    benchmark = _MockBenchmark()
    layer_selection = LayerSelection(model_identifier=model_identifier,
                                     activations_model=model_instance, layers=layers,
                                     visual_degrees=8)
    region_layer_map = RegionLayerMap(layer_selection=layer_selection,
                                      region_benchmarks={'IT': benchmark})

    brain_model = ModelCommitment(identifier=model_identifier, activations_model=model_instance,
                                  layers=layers, region_layer_map=region_layer_map)
    score = benchmark(brain_model, do_behavior=True)
    assert score is not None


class _MockBenchmark(BenchmarkBase):
    def __init__(self):
        assembly_repetition = load_dataset("MajajHong2015.public").sel(region="IT").squeeze("time_bin")
        assert hasattr(assembly_repetition, 'repetition')
        self.region = 'IT'
        self.assembly = average_repetition(assembly_repetition)
        self._assembly = self.assembly
        self.timebins = timebins_from_assembly(self.assembly)
        self._similarity_metric = load_metric('pls', crossvalidation_kwargs=dict(stratification_coord='object_name'))
        identifier = f'{assembly_repetition.name}-layer_selection'
        ceiler = load_ceiling('internal_consistency')
        super(_MockBenchmark, self).__init__(identifier=identifier,
                                             ceiling_func=lambda: ceiler(assembly_repetition),
                                             version='1.0')

    def __call__(self, candidate: BrainModel, do_behavior=False):
        # adapt stimuli to visual degrees
        stimuli = place_on_screen(self.assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                  source_visual_degrees=8)  # arbitrary choice for source degrees
        # Check neural recordings
        candidate.start_recording(self.region, time_bins=self.timebins)
        source_assembly = candidate.look_at(stimuli)
        # Check behavioral tasks
        if do_behavior:
            candidate.start_task(BrainModel.Task.probabilities, self.assembly.stimulus_set)
            candidate.look_at(stimuli)
        raw_score = self._similarity_metric(source_assembly, self.assembly)
        return ceil_score(raw_score, self.ceiling)

