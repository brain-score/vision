import logging

from result_caching import store_xarray, store
from tqdm import tqdm
from typing import Optional, Dict, List

from brainscore_vision.metrics import Score
from brainscore_vision.model_helpers.activations.pca import LayerPCA
from brainscore_vision.model_helpers.brain_transformation import TemporalIgnore
from brainscore_vision.model_helpers.utils import make_list
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import fullname


class LayerMappedModel(BrainModel):
    def __init__(self, identifier, activations_model, region_layer_map, visual_degrees=None):
        self._identifier = identifier
        self.activations_model = activations_model
        self._visual_degrees = visual_degrees
        self.region_layer_map = region_layer_map
        self.recorded_regions = []

    @property
    def identifier(self):
        return self._identifier

    def look_at(self, stimuli, number_of_trials=1, require_variance: bool = False):
        """
        TODO
        :param number_of_trials: An integer that determines how many repetitions of the same model performs.
        :param require_variance: A bool that asks models to output different responses to the same stimuli (i.e.,
            allows stochastic responses to identical stimuli, even in deterministic models). The current implementation
            implements this using microsaccades.
            Human microsaccade amplitude varies by who you ask, an estimate might be <0.1 deg = 360 arcsec = 6arcmin.
            The goal of microsaccades is to obtain multiple different neural activities to the same input stimulus
            from non-stochastic models. This is to improve estimates of e.g. psychophysical functions, but also other
            things. Note that microsaccades are also applied to stochastic models to make them comparable within-
            benchmark to non-stochastic models.
            In the current implementation, if `require_variance=True`, the model selects microsaccades according to
            its own microsaccade behavior (if it has implemented it), or with the base behavior of saccading in
            input pixel space with 1-pixel increments from the center of the stimulus. The base behavior thus
            maintains a fixed microsaccade distance as measured in visual angle, regardless of the model's visual angle.
            Example usage:
                require_variance = True
            More information:
            --> Rolfs 2009 "Microsaccades: Small steps on a long way" Vision Research, Volume 49, Issue 20, 15
            October 2009, Pages 2415-2441.
            --> Haddad & Steinmann 1973 "The smallest voluntary saccade: Implications for fixation" Vision
            Research Volume 13, Issue 6, June 1973, Pages 1075-1086, IN5-IN6.
            Thanks to Johannes Mehrer for initial help in implementing microsaccades.
        """
        layer_regions = {}
        for region in self.recorded_regions:
            layers = self.region_layer_map[region]
            layers = make_list(layers)
            for layer in layers:
                assert layer not in layer_regions, f"layer {layer} has already been assigned for {layer_regions[layer]}"
                layer_regions[layer] = region
        activations = self.run_activations(stimuli,
                                           layers=list(layer_regions.keys()),
                                           number_of_trials=number_of_trials,
                                           require_variance=require_variance)
        activations['region'] = 'neuroid', [layer_regions[layer] for layer in activations['layer'].values]
        return activations

    def run_activations(self, stimuli, layers, number_of_trials=1, require_variance=None):
        activations = self.activations_model(stimuli, layers=layers, number_of_trials=number_of_trials,
                                             require_variance=require_variance)
        return activations

    def start_task(self, task):
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target: BrainModel.RecordingTarget):
        self.recorded_regions = [recording_target]

    def visual_degrees(self) -> int:
        return self._visual_degrees


class LayerSelection:
    def __init__(self, model_identifier, activations_model, layers, visual_degrees):
        """
        :param model_identifier: this is separate from the container name because it only refers to
            the combination of (model, preprocessing), i.e. no mapping.
        """
        self.model_identifier = model_identifier
        self._layer_scoring = LayerScores(model_identifier=model_identifier, activations_model=activations_model,
                                          visual_degrees=visual_degrees)
        self.layers = layers
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, selection_identifier, benchmark):
        # for layer-mapping, attach LayerPCA so that we can cache activations
        model_identifier = self.model_identifier
        pca_hooked = LayerPCA.is_hooked(self._layer_scoring._activations_model)
        if not pca_hooked:
            pca_handle = LayerPCA.hook(self._layer_scoring._activations_model, n_components=1000)
            identifier = self._layer_scoring._activations_model.identifier
            self._layer_scoring._activations_model.identifier = identifier + "-pca_1000"
            model_identifier += "-pca_1000"

        result = self._call(model_identifier=model_identifier, selection_identifier=selection_identifier,
                            benchmark=benchmark)

        if not pca_hooked:
            pca_handle.remove()
            self._layer_scoring._activations_model.identifier = identifier
        return result

    @store(identifier_ignore=['benchmark', 'benchmark'])
    def _call(self, model_identifier, selection_identifier, benchmark):
        self._logger.debug("Finding best layer")
        layer_scores = self._layer_scoring(benchmark=benchmark, benchmark_identifier=selection_identifier,
                                           layers=self.layers, prerun=True)

        self._logger.debug("Layer scores (unceiled): " + ", ".join([
            f"{layer} -> {layer_scores.raw.sel(layer=layer).item():.3f}"
            f"+-{layer_scores.raw.sel(layer=layer).attrs['error'].item():.3f}"
            for layer in layer_scores['layer'].values]))
        best_layer = layer_scores['layer'].values[layer_scores.argmax()]
        return best_layer


class LayerScores:
    def __init__(self, model_identifier, activations_model, visual_degrees):
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self._visual_degrees = visual_degrees
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, benchmark, layers, benchmark_identifier=None, prerun=False):
        return self._call(model_identifier=self.model_identifier,
                          benchmark_identifier=benchmark_identifier or benchmark.identifier,
                          visual_degrees=self._visual_degrees,
                          model=self._activations_model, benchmark=benchmark, layers=layers, prerun=prerun)

    @store_xarray(identifier_ignore=['model', 'benchmark', 'layers', 'prerun'], combine_fields={'layers': 'layer'})
    def _call(self, model_identifier, benchmark_identifier, visual_degrees,  # storage fields
              model, benchmark, layers, prerun=False):
        layer_scores = []
        for i, layer in enumerate(tqdm(layers, desc="layers")):
            layer_model = self._create_mapped_model(region=benchmark.region, layer=layer, model=model,
                                                    model_identifier=model_identifier, visual_degrees=visual_degrees)
            layer_model = TemporalIgnore(layer_model)
            if i == 0 and prerun:  # pre-run activations together to avoid running every layer separately
                # we can only pre-run stimuli in response to the benchmark, since we might otherwise be missing
                # visual_degrees resizing.
                layer_model = PreRunLayers(model=model, layers=layers, forward=layer_model)
            score = benchmark(layer_model)
            score = score.expand_dims('layer')
            score['layer'] = [layer]
            layer_scores.append(score)
        layer_scores = Score.merge(*layer_scores)
        layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
        return layer_scores

    def _create_mapped_model(self, region, layer, model, model_identifier, visual_degrees):
        return LayerMappedModel(identifier=f"{model_identifier}-{layer}", visual_degrees=visual_degrees,
                                # per-layer identifier to avoid overlap
                                activations_model=model, region_layer_map={region: layer})


class PreRunLayers:
    def __init__(self, model, layers, forward):
        self._model = model
        self._layers = layers
        self._forward = forward

    def look_at(self, stimuli, number_of_trials=1):
        self._model(layers=self._layers, stimuli=stimuli)
        return self._forward.look_at(stimuli, number_of_trials=number_of_trials)

    def __getattr__(self, item):
        if item in ['look_at']:
            return super(PreRunLayers, self).__getattr__(item)
        return getattr(self._forward, item)
