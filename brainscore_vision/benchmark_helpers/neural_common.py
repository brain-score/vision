import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
import itertools

from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly, array_is_element, walk_coords
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
    """
    Neural benchmark with separate train and test assemblies.
    
    similarity_metric must be a metric such as the regression_correlation metrics named 
    '[regression_type]-split' (e.g. 'ridgecv-split') which take four arguments: 
    -> source and target assemblies to fit the mapping 
    -> separate source and tagets assemblies to evaluate.
    
    Parameter alpha_coord can be used to specficy an assembly coordinate, where
    unique values should be fitted separately, e.g. use alpha_coord='subject' to fit an individual
    ridgecv alpha for each subject.

    If per_voxel_ceilings=True ceilings are applied to neuroids before aggregating, otherwise afterwards (default).
    """
    
    def __init__(self, identifier, ceiling_func, version, 
                 train_assembly, test_assembly, similarity_metric, 
                 visual_degrees, number_of_trials, parent,
                 alpha_coord: str=None, 
                 per_voxel_ceilings: bool=False,
                 **kwargs):
        
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

        self.timebins = timebins_from_assembly(self.train_assembly)
        
        self.alpha_coord = alpha_coord # fit a separate mapping for each unique value along this coord

        if per_voxel_ceilings:
            self.ceiling_mode = neuroid_wise_explained_var
        else:
            self.ceiling_mode = explained_variance
        
    def __call__(self, candidate: BrainModel):  
        """
        Score a candidate model on this benchmark.
        
        Returns
        -------
        Score
            Score relative to ceiling. 
            If `alpha_coord` is set, results for each unique coord value are stored as attributes:
            -> score.values: the final ceiled score (mean of all individually fitted slices, e.g. subjects)
            -> score.raw: the mean of all the raw values
            -> score.celing: aggregate ceiling value of the benchmark 
            Note: we aggregate neuroids for each slice and then ceil the slice, meaning the overall ceiling is not applied directly
            score.attrs[alpha_coord_value] will contain the standard score object with per neuroid raw and ceiling values
        """
        
        # get the activations from the train set
        train_stimulus_set = self.train_assembly.stimulus_set
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(train_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
        self.train_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        # get the activations from the test set
        test_stimulus_set = self.test_assembly.stimulus_set
        timebins = timebins_from_assembly(self.test_assembly)
        candidate.start_recording(self.region, time_bins=self.timebins)  
        stimulus_set = place_on_screen(test_stimulus_set, target_visual_degrees=candidate.visual_degrees(),
									source_visual_degrees=self._visual_degrees)
        self.test_activations = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        # squeeze time_bin dim if it has length one
        # regression only supports (presentation x neuroids) arrays, but temporal models like CORnet return a time_bin dim
        if 'time_bin' in self.train_activations.dims and self.train_activations.sizes['time_bin'] == 1:
            self.train_activations = self.train_activations.squeeze('time_bin')
        if 'time_bin' in self.test_activations.dims and self.test_activations.sizes['time_bin'] == 1:
            self.test_activations = self.test_activations.squeeze('time_bin')

        if self.alpha_coord is not None:
            scores_dict = {}
            alpha_splits = np.unique(self.train_assembly[self.alpha_coord].values)
            for coord_value in alpha_splits:
                coord_dict = {self.alpha_coord: coord_value}
                train_subset = select_with_preserved_index(self.train_assembly, coord_dict)
                test_subset = select_with_preserved_index(self.test_assembly, coord_dict)
                
                # calculate the ceiling for the subset, based on only those per-neuroid ceilings contained in the slice
                raw_ceilings_slice = select_with_preserved_index(self.ceiling.raw, coord_dict)
                # recalculate median of neuroids, mean of cv-splits as done by CrossValidationSingle
                subset_ceiling = Score(np.mean(np.median(raw_ceilings_slice, axis=1)))
                subset_ceiling.attrs['raw'] = raw_ceilings_slice
                
                score = self.get_score(train_data=train_subset, 
                                       test_data=test_subset,
                                       ceiling_values=subset_ceiling,
                                       apply_ceiling=self.ceiling_mode)
                score = score.expand_dims(self.alpha_coord)
                score[self.alpha_coord] = [coord_value]
                print(score)
                scores_dict[coord_value] = score
            
            # the score is the mean of all the individual ceiled scores:
            score = Score(np.mean([s.values for s in scores_dict.values()]))
            
            # the overall raw value is the mean of raw values (individual values are thus found in .raw.raw)
            score.attrs[Score.RAW_VALUES_KEY] = Score(np.mean([s.raw.values for s in scores_dict.values()]))
            score.attrs[Score.RAW_VALUES_KEY].attrs[Score.RAW_VALUES_KEY] = xr.concat(scores_dict.values(), 
                                                                                      dim=self.alpha_coord, 
                                                                                      combine_attrs='drop')
            
            # the ceiling is the overall ceiling aggregated across all neuroids irrespective of splits
            score.attrs['ceiling'] = self.ceiling
            
            # each individual alpha that was fitted is documented as an attribute 
            # (including raw values and ceiling for that split)
            for coord_value in alpha_splits:
                score.attrs[coord_value] = scores_dict[coord_value]
            return score

        else:
            score = self.get_score(train_data=self.train_assembly, 
                                   test_data=self.test_assembly,
                                   ceiling_values=self.ceiling, 
                                   apply_ceiling=self.ceiling_mode)
            return score

    def get_score(self, train_data, test_data, ceiling_values, apply_ceiling):
        """
        train_data : NeuroidAssembly, neural recordings to fit
        test_data : NeuroidAssembly, neural recordings to predict
        ceiling_values : DataArray, the ceiling values for each neuroid
        apply_ceiling : function, method how to apply ceilings (per neuroid or overall)
        """
        raw_score = self._similarity_metric(source_train=self.train_activations,
                                            target_train=train_data,
                                            source_test=self.test_activations,
                                            target_test=test_data)
        ceiled_score = apply_ceiling(raw_score, ceiling_values)
        return ceiled_score


def select_with_preserved_index(assembly, coord_dict):
    new_assembly = assembly.sel(neuroid=coord_dict, drop=False)
    if new_assembly.sizes['neuroid'] == 1:
        # reassign dropped coordinates from coord dict (for each key, set coordinate key to value coord_dict[key])
        new_assembly = new_assembly.assign_coords({key: ('neuroid', [coord_dict[key]]) for key in coord_dict})
        new_assembly = new_assembly.reset_index('neuroid')
        new_assembly = new_assembly.set_index(neuroid=assembly.get_index('neuroid').names)
    new_assembly = new_assembly.transpose(*assembly.dims)
    return new_assembly


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

def neuroid_wise_explained_var(score: Score, ceiling: Score, aggregate_func=np.mean) -> Score:
    """
    Computes the explained variance for every neuroid individually.
    Parameters
    ----------
    score : Score
        The raw score containing neuroid-wise correlations.
    ceiling : Score
        The ceiling containing neuroid-wise ceilings.
    Returns
    -------
    ceiled_score : Score
        The aggregate neuroid-wise explained variance score.
        .attrs['ceiled_scores'] contains the individual neuroids' ceiled scores.
    """
    assert score.raw is not None, "Score must have raw values for neuroid-wise explained variance"
    raw_scores = score.raw
    
    if hasattr(ceiling, 'raw'): 
        raw_ceilings = ceiling.raw  # ceiling provided with aggregate and raw values
    else:
        raw_ceilings = ceiling      # ceiling was not yet aggregated and can be accessed as is
     
    if 'split' in raw_ceilings.dims:
        raw_ceilings = raw_ceilings.mean(dim='split')  # averaging ceiling across splits if applicable
    assert raw_scores.dims == raw_ceilings.dims, "score and ceiling dims don't match"
    
    # assert perfect coordinate alignment between raws and ceilings
    pd.testing.assert_index_equal(raw_ceilings.indexes['neuroid'], raw_scores.indexes['neuroid']) 

    # apply explained variance neuroid-wise
    r_square_neuroids = np.power(raw_scores / raw_ceilings, 2)
    ceiled_score = aggregate_func(r_square_neuroids)  # aggregate across neuroids
    ceiled_score = Score(ceiled_score)
    ceiled_score.attrs['ceiled_scores'] = r_square_neuroids
    ceiled_score.attrs['ceiling'] = ceiling
    
    # attach all previous attributes, prevent overwriting
    for key, value in score.attrs.items():
        # attaches the raw scores and other info of the original score
        # Note: if score already had a ceiling attribute, it will be dropped
        if key not in ceiled_score.attrs:
            ceiled_score.attrs[key] = value
    for key, value in score.raw.attrs.items():
        # propagate attributes from raw score if not already present (e.g. ridge cv alpha)
        if key not in ceiled_score.attrs:
            ceiled_score.attrs[key] = value
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

    assert assembly.dims == ('presentation', 'neuroid', 'time_bin'), \
        f"Flattening assembly expects dims (presentation, neuroid, time_bin), got {assembly.dims}"
    assert 'time_bin_start' in assembly.indexes['time_bin'].names, \
        "Expected 'time_bin_start' coordinate in time_bin dimension"
    assert 'time_bin_end' in assembly.indexes['time_bin'].names, \
        "Expected 'time_bin_end' coordinate in time_bin dimension"
    
    # flatten the data
    n_presentations, n_neuroids, n_timebins = assembly.shape
    attributes = assembly.attrs
    flattened_data = assembly.data.reshape(
        n_presentations,
        n_neuroids * n_timebins,
        1
    )

    # expand the presentation dim x n_timebins
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

def filter_reliable_neuroids(assembly : DataAssembly, reliab_threshold : float, reliab_coord : str) -> DataAssembly:
    """
    Filter out neuroids below a reliability threshold.
    Using DataArray.where returns a DataArray object so we need to reconstruct the original NeuroidAssembly (or related) class.
    """
    assembly_type = type(assembly)
    filtered_assembly = assembly.where(assembly[reliab_coord] > reliab_threshold, drop=True)
    return assembly_type(filtered_assembly)