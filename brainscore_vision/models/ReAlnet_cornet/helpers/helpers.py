import re
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from brainio.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation.behavior import BehaviorArbiter, LogitsBehavior, \
    ProbabilitiesMapping, OddOneOut
from brainscore_vision.model_interface import BrainModel
from result_caching import store


class TemporalPytorchWrapper(PytorchWrapper):
    def __init__(self, *args, separate_time=True, **kwargs):
        self._separate_time = separate_time
        super(TemporalPytorchWrapper, self).__init__(*args, **kwargs)

    def _build_extractor(self, *args, **kwargs):
        if self._separate_time:
            return TemporalExtractor(*args, **kwargs)
        else:
            return super(TemporalPytorchWrapper, self)._build_extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        # reset
        self._layer_counter = defaultdict(lambda: 0)
        self._layer_hooks = {}
        return super(TemporalPytorchWrapper, self).get_activations(images=images, layer_names=layer_names)

    def register_hook(self, layer, layer_name, target_dict):
        layer_name = self._strip_layer_timestep(layer_name)
        if layer_name in self._layer_hooks:  # add hook only once for multiple timesteps
            return self._layer_hooks[layer_name]

        def hook_function(_layer, _input, output):
            target_dict[f"{layer_name}-t{self._layer_counter[layer_name]}"] = PytorchWrapper._tensor_to_numpy(output)
            self._layer_counter[layer_name] += 1

        hook = layer.register_forward_hook(hook_function)
        self._layer_hooks[layer_name] = hook
        return hook

    def get_layer(self, layer_name):
        layer_name = self._strip_layer_timestep(layer_name)
        return super(TemporalPytorchWrapper, self).get_layer(layer_name)

    def _strip_layer_timestep(self, layer_name):
        match = re.search('-t[0-9]+$', layer_name)
        if match:
            layer_name = layer_name[:match.start()]
        return layer_name


class CORnetCommitment(BrainModel):
    """
    CORnet commitment where only the model interface is implemented and behavioral readouts are attached.
    Importantly, layer-region commitments do not occur due to the anatomical pre-mapping.
    Further, due to the temporal component of the model, requested time-bins are matched to the nearest committed
    time-bin for the model.
    """

    def __init__(self, identifier, activations_model, layers,
                 time_mapping: Dict[str, Dict[int, Tuple[int, int]]], behavioral_readout_layer=None,
                 visual_degrees=8):
        """
        :param time_mapping: mapping from region -> {model_timestep -> (time_bin_start, time_bin_end)}
        """
        self.layers = layers
        self.activations_model = activations_model
        self.time_mapping = time_mapping
        self.recording_layers = None
        self.recording_time_bins = None
        self._identifier = identifier

        logits_behavior = LogitsBehavior(
            identifier=identifier, activations_model=TemporalIgnore(activations_model))
        behavioral_readout_layer = behavioral_readout_layer or layers[-1]
        probabilities_behavior = ProbabilitiesMapping(
            identifier=identifier, activations_model=TemporalIgnore(activations_model), layer=behavioral_readout_layer)
        odd_one_out = OddOneOut(identifier=identifier, activations_model=TemporalIgnore(activations_model),
                                layer=behavioral_readout_layer)
        self.behavior_model = BehaviorArbiter({BrainModel.Task.label: logits_behavior,
                                               BrainModel.Task.probabilities: probabilities_behavior,
                                               BrainModel.Task.odd_one_out: odd_one_out,
                                               })
        self.do_behavior = False

        self._visual_degrees = visual_degrees

    @property
    def identifier(self):
        return self._identifier

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_recording(self, recording_target, time_bins):
        self.recording_layers = [layer for layer in self.layers if layer.startswith(recording_target)]
        self.recording_time_bins = time_bins

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        if task != BrainModel.Task.passive:
            self.behavior_model.start_task(task, *args, **kwargs)
            self.do_behavior = True

    def look_at(self, stimuli, number_of_trials: int = 1, require_variance: bool = False):
        if self.do_behavior:
            return self.behavior_model.look_at(stimuli,
                                               number_of_trials=number_of_trials, require_variance=require_variance)
        else:
            # cache, since piecing times together is not too fast unfortunately
            return self.look_at_cached(self.identifier, stimuli.identifier, stimuli,
                                       number_of_trials=number_of_trials, require_variance=require_variance)

    @store(identifier_ignore=['stimuli', 'number_of_trials', 'require_variance'])
    def look_at_cached(self, model_identifier, stimuli_identifier, stimuli,
                       number_of_trials, require_variance):
        responses = self.activations_model(stimuli, layers=self.recording_layers,
                                           number_of_trials=number_of_trials, require_variance=require_variance)
        # map time
        regions = set(responses['region'].values)
        if len(regions) > 1:
            raise NotImplementedError("cannot handle more than one simultaneous region")
        region = list(regions)[0]
        time_bins = [self.time_mapping[region][timestep] if timestep in self.time_mapping[region] else (None, None)
                     for timestep in responses['time_step'].values]
        responses['time_bin_start'] = 'time_step', [time_bin[0] for time_bin in time_bins]
        responses['time_bin_end'] = 'time_step', [time_bin[1] for time_bin in time_bins]
        responses = NeuroidAssembly(responses.rename({'time_step': 'time_bin'}))
        responses = responses[{'time_bin': [not np.isnan(time_start) for time_start in responses['time_bin_start']]}]
        # select time
        time_responses = []
        for time_bin in tqdm(self.recording_time_bins, desc='CORnet-time to recording time'):
            time_bin = time_bin if not isinstance(time_bin, np.ndarray) else time_bin.tolist()
            time_bin_start, time_bin_end = time_bin
            nearest_start = find_nearest(responses['time_bin_start'].values, time_bin_start)
            bin_responses = responses.sel(time_bin_start=nearest_start)
            bin_responses = NeuroidAssembly(bin_responses.values, coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(bin_responses)
                   if coord not in ['time_bin_level_0', 'time_bin_end']},
                **{'time_bin_start': ('time_bin', [time_bin_start]),
                   'time_bin_end': ('time_bin', [time_bin_end])}
            }, dims=bin_responses.dims)
            time_responses.append(bin_responses)
        responses = merge_data_arrays(time_responses)
        return responses


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class TemporalIgnore:
    """
    Wrapper around a activations model that squeezes out the temporal axis.
    Useful when there is only one time step and the behavioral readout does not know what to do with time.
    """

    def __init__(self, temporal_activations_model):
        self._activations_model = temporal_activations_model

    def __call__(self, *args, **kwargs):
        activations = self._activations_model(*args, **kwargs)
        activations = activations.squeeze('time_step')
        return activations


class TemporalExtractor(ActivationsExtractorHelper):
    # `from_paths` is the earliest method at which we can interject because calls below are stored and checked for the
    # presence of all layers which, for CORnet, are passed as e.g. `IT.output-t0`.
    # This code re-arranges the time component.
    def from_paths(self, *args, **kwargs):
        raw_activations = super(TemporalExtractor, self).from_paths(*args, **kwargs)
        # introduce time dimension
        regions = defaultdict(list)
        for layer in set(raw_activations['layer'].values):
            match = re.match(r'(([^-]*)\..*|logits|avgpool)-t([0-9]+)', layer)
            region, timestep = match.group(2) if match.group(2) else match.group(1), match.group(3)
            stripped_layer = match.group(1)
            regions[region].append((layer, stripped_layer, timestep))
        activations = {}
        for region, time_layers in regions.items():
            for (full_layer, stripped_layer, timestep) in time_layers:
                region_time_activations = raw_activations.sel(layer=full_layer)
                region_time_activations['layer'] = 'neuroid', [stripped_layer] * len(region_time_activations['neuroid'])
                activations[(region, timestep)] = region_time_activations
        for key, key_activations in activations.items():
            region, timestep = key
            key_activations['region'] = 'neuroid', [region] * len(key_activations['neuroid'])
            activations[key] = NeuroidAssembly([key_activations.values], coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(activations[key])
                   if coord != 'neuroid_id'},  # otherwise, neuroid dim will be as large as before with nans
                **{'time_step': [int(timestep)]}
            }, dims=['time_step'] + list(key_activations.dims))
        activations = list(activations.values())
        activations = merge_data_arrays(activations)
        # rebuild neuroid_id without timestep
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            activations[coord].values for coord in ['model', 'region', 'neuroid_num']])]
        activations['neuroid_id'] = 'neuroid', neuroid_id
        return activations


def _build_time_mappings(time_mappings):
    return {region: {
        timestep: (time_start + timestep * time_step_size,
                   time_start + (timestep + 1) * time_step_size)
        for timestep in range(0, timesteps)}
        for region, (time_start, time_step_size, timesteps) in time_mappings.items()}
