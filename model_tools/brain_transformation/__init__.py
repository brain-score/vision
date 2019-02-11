from bidict import bidict
import logging

import numpy as np
from tqdm import tqdm

from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import average_repetition
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
from result_caching import store_xarray, store


class ModelCommitment(BrainModel):
    def __init__(self, identifier, base_model, layers):
        self.identifier = identifier
        self.base_model = base_model
        self.layers = layers
        self.region_layer = bidict()
        self.recorded_regions = []

    def commit_region(self, region, commitment_data):
        layer_selection = LayerSelection(model_identifier=self.identifier,
                                         activations_model=self.base_model, layers=self.layers)
        layer = layer_selection(commitment_data)
        self.region_layer[region] = layer

    def look_at(self, stimuli):
        activations = self.base_model(stimuli, layers=[self.region_layer[region] for region in self.recorded_regions])
        activations['region'] = 'neuroid', [self.region_layer.inv[layer] for layer in activations['layer'].values]
        return activations

    def start_task(self, task):
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target):
        if str(recording_target) not in self.region_layer:
            raise NotImplementedError(f"Region {recording_target} is not committed")
        self.recorded_regions.append(recording_target)


class LayerModel(BrainModel):
    def __init__(self, identifier, base_model, layer, region):
        self.identifier = identifier
        self.base_model = base_model
        self.layer = layer
        self.region = region

    def look_at(self, stimuli):
        activations = self.base_model(stimuli, layers=[self.layer])
        activations['region'] = 'neuroid', np.repeat([self.region], len(activations['layer']))
        return activations

    def start_task(self, task):
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target):
        if str(recording_target) != self.region:
            raise NotImplementedError(f"Region {recording_target} is not available")


class LayerSelection:
    def __init__(self, model_identifier, activations_model, layers):
        """
        :param model_identifier: this is separate from the container name because it only refers to
            the combination of (model, preprocessing), i.e. no mapping.
        """
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self.layers = layers
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly):
        return self._call(model_identifier=self.model_identifier, assembly_identifier=assembly.name, assembly=assembly)

    @store(identifier_ignore=['assembly'])
    def _call(self, model_identifier, assembly_identifier, assembly):
        benchmark = self.build_benchmark(assembly)
        self._logger.debug("Finding best layer")
        layer_scores = self._layer_scores(
            model_identifier=self.model_identifier, layers=self.layers,
            benchmark_identifier=assembly_identifier, benchmark=benchmark)
        self._logger.debug(f"Layer scores (unceiled): {layer_scores.raw}")
        best_layer = layer_scores['layer'].values[layer_scores.sel(aggregation='center').argmax()]
        return best_layer

    @store_xarray(identifier_ignore=['benchmark', 'layers'], combine_fields={'layers': 'layer'})
    def _layer_scores(self, model_identifier, layers, benchmark_identifier,  # storage fields
                      benchmark):
        # pre-run activations together to avoid running every layer separately
        self._activations_model(layers=self.layers, stimuli=benchmark.assembly.stimulus_set)

        layer_scores = []
        for layer in tqdm(self.layers):
            layer_model = LayerModel(identifier=model_identifier, base_model=self._activations_model,
                                     layer=layer, region=benchmark.region)
            score = benchmark(layer_model)
            score = score.expand_dims('layer')
            score['layer'] = [layer]
            layer_scores.append(score)
        layer_scores = Score.merge(*layer_scores)
        return layer_scores

    class _Benchmark(BenchmarkBase):
        def __init__(self, assembly_repetition, similarity_metric=None, ceiler=None):
            # assert len(np.unique(assembly_repetition['region'])) == 1
            # self.region = np.unique(assembly_repetition['region'])[0]
            self.region = 'layer_selection-dummy_region'
            self.assembly = average_repetition(assembly_repetition)

            self._similarity_metric = similarity_metric or CrossRegressedCorrelation()
            name = f'{assembly_repetition.name}-layer_selection'
            ceiler = ceiler or InternalConsistency()
            super(LayerSelection._Benchmark, self).__init__(
                name=name, ceiling_func=lambda: ceiler(assembly_repetition))

        def __call__(self, candidate):
            candidate.start_recording(self.region)
            source_assembly = candidate.look_at(self.assembly.stimulus_set)
            raw_score = self._similarity_metric(source_assembly, self.assembly)
            return ceil_score(raw_score, self.ceiling)

    def build_benchmark(self, assembly):
        return self._Benchmark(assembly)


def single_element(element_list):
    assert len(element_list) == 1
    return element_list[0]
