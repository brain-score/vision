import logging
from typing import Optional

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
        self.layers = layers

        self.layer_model = LayerModel(identifier=identifier, base_model=base_model)
        # forward brain-interface methods
        self.look_at = self.layer_model.look_at
        self.start_recording = self.layer_model.start_recording
        self.start_task = self.layer_model.start_task

    def commit_region(self, region, assembly):
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.base_model, layers=self.layers)
        best_layer = layer_selection(assembly)
        self.layer_model.commit(region, best_layer)


class LayerModel(BrainModel):
    def __init__(self, identifier, base_model, region_layer_map: Optional[dict] = None):
        self.identifier = identifier
        self.base_model = base_model
        self.region_layer_map = region_layer_map or {}
        self.recorded_regions = []

    def look_at(self, stimuli):
        layer_regions = {self.region_layer_map[region]: region for region in self.recorded_regions}
        assert len(layer_regions) == len(self.recorded_regions), f"duplicate layers for {self.recorded_regions}"
        activations = self.base_model(stimuli, layers=list(layer_regions.keys()))
        activations['region'] = 'neuroid', [layer_regions[layer] for layer in activations['layer'].values]
        return activations

    def start_task(self, task):
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target):
        if str(recording_target) not in self.region_layer_map:
            raise NotImplementedError(f"Region {recording_target} is not committed")
        self.recorded_regions = [recording_target]

    def commit(self, region, layer):
        self.region_layer_map[region] = layer


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
        benchmark = self._Benchmark(assembly)
        self._logger.debug("Finding best layer")
        layer_scores = self._layer_scores(
            model_identifier=self.model_identifier, layers=self.layers,
            benchmark_identifier=assembly_identifier, benchmark=benchmark)
        self._logger.debug("Layer scores (unceiled): " + ", ".join([
            f"{layer} -> {layer_scores.raw.sel(layer=layer, aggregation='center').values:.3f}"
            f"+-{layer_scores.raw.sel(layer=layer, aggregation='error').values:.3f}"
            for layer in layer_scores['layer'].values]))
        best_layer = layer_scores['layer'].values[layer_scores.sel(aggregation='center').argmax()]
        return best_layer

    @store_xarray(identifier_ignore=['benchmark', 'layers'], combine_fields={'layers': 'layer'})
    def _layer_scores(self, model_identifier, layers, benchmark_identifier,  # storage fields
                      benchmark):
        # pre-run activations together to avoid running every layer separately
        self._activations_model(layers=layers, stimuli=benchmark.assembly.stimulus_set)

        layer_scores = []
        for layer in tqdm(layers, desc="layers"):
            layer_model = LayerModel(identifier=model_identifier, base_model=self._activations_model,
                                     region_layer_map={benchmark.region: layer})
            score = benchmark(layer_model)
            score = score.expand_dims('layer')
            score['layer'] = [layer]
            layer_scores.append(score)
        layer_scores = Score.merge(*layer_scores)
        layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
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


def single_element(element_list):
    assert len(element_list) == 1
    return element_list[0]
