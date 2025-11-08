import numpy as np
import pandas as pd
from xarray import DataArray

from brainscore_core.supported_data_standards.brainio.assemblies import array_is_element, walk_coords
from brainscore_core import Score
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel
from .screen import place_on_screen


class NeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, visual_degrees, number_of_trials, **kwargs):
        super(NeuralBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        if 'time_bin' in source_assembly.dims and source_assembly.sizes['time_bin'] == 1:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        ceiled_score = explained_variance(raw_score, self.ceiling)
        return ceiled_score

class TrainTestNeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, ceiling_func, version, 
                 train_assembly, test_assembly, similarity_metric, 
                 visual_degrees, number_of_trials, parent, **kwargs):
        super(TrainTestNeuralBenchmark, self).__init__(identifier=identifier, ceiling_func=ceiling_func,
                                                version=version, parent=parent, **kwargs)
        self.train_assembly = train_assembly
        self.test_assembly = test_assembly
        self._similarity_metric = similarity_metric
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        
        region = np.unique(self.train_assembly['region'])
        assert len(region) == 1
        assert region[0] == np.unique(self.test_assembly['region'])[0]
        self.region = region[0]

        timebins = timebins_from_assembly(self.train_assembly)
        self.timebins = timebins
        
    def __call__(self, candidate: BrainModel):  
        
        # get the activations from the train set
        train_stimulus_set = self.train_assembly.stimulus_set
        timebins = timebins_from_assembly(self.train_assembly)
        candidate.start_recording(self.region, time_bins=timebins)
        stimulus_set = place_on_screen(train_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
        train_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        # get the activations from the test set
        test_stimulus_set = self.test_assembly.stimulus_set
        timebins = timebins_from_assembly(self.test_assembly)
        candidate.start_recording(self.region, time_bins=timebins)
        stimulus_set = place_on_screen(test_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
									source_visual_degrees=self._visual_degrees)
        test_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        raw_score = self._similarity_metric(source_train=train_activations, source_test=test_activations,
                target_train=self.train_assembly, target_test=self.test_assembly)
        ceiled_score = explained_variance(raw_score, self.ceiling)
        return ceiled_score


def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins


def explained_variance(score: Score, ceiling: Score) -> Score:
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(score.values /
                        ceiling.values, 2)
    ceiled_score = Score(r_square)
    if 'error' in score.attrs:
        ceiled_score.attrs['error'] = score.attrs['error']
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score


def average_repetition(assembly):
    def avg_repr(assembly):
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation') and coord != 'repetition']
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        return assembly

    return apply_keep_attrs(assembly, avg_repr)

def apply_keep_attrs(assembly, fnc):  # workaround to keeping attrs
    attrs = assembly.attrs
    assembly = fnc(assembly)
    assembly.attrs = attrs
    return assembly

def flatten_timebins_into_neuroids(assembly: DataArray) -> DataArray:
    """
    For data with multiple time bins, flatten the time bins into the neuroid dimension.
    Facilitates data handling for benchmarks that predict individual time bins as separate neuroids, without considering their relative position in time.
    Adds dimension time_bin to neuroid dimension, that contains the start of the time_bin.
    Workaround cause xarray cannot deal with stacking MultiIndex (pydata/xarray#1554), see also metric_helpers/temporal.py

    :param assembly: DataArray of shape (presentation, neuroid, time_bin)
    :return flattened_assembly: DataArray of shape (presentation, neuroid * time_bin, 1)

    """

    #flatten the data
    n_presentations, n_neuroids, n_timebins = assembly.shape
    attributes = assembly.attrs
    flattened_data = assembly.data.reshape(
        n_presentations,
        n_neuroids * n_timebins,
        1
    )

    #expand the presentation dim x n_timebins
    coords = {k: v for k, v in assembly.coords.items() if k != "time_bin"}
    old_index = coords['neuroid'].to_index()
    coords['neuroid'] = pd.MultiIndex.from_tuples(
    np.repeat(old_index.to_numpy(), n_timebins),
    names=old_index.names
    )
    time_bin_start = assembly.coords['time_bin_start']
    assert len(time_bin_start) == n_timebins, "time_bin_start length does not match number of time bins"
    recoding_times = np.tile(time_bin_start, reps=assembly.shape[1])
    window_start = assembly.coords['time_bin_start'].values[0]
    window_end = assembly.coords['time_bin_end'].values[-1]

    flattened_assembly = type(assembly)(
        flattened_data,
        dims=assembly.dims,
        coords={
                "recoding_time": ("neuroid", recoding_times),
                "time_bin_start": ("time_bin", [window_start]),
                "time_bin_end": ("time_bin", [window_end]),
                **coords
        },
    )
    flattened_assembly.attrs = attributes
    return flattened_assembly