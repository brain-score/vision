import numpy as np
import os
import pandas as pd
import xarray as xr

from brainio_base.assemblies import array_is_element, walk_coords
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from brainscore.benchmarks._neural_common import explained_variance, timebins_from_assembly
from brainio_base.stimuli import StimulusSet
from brainscore.metrics.regression_extra import take_gram, unflatten


class NeuralBenchmarkCovariate(BenchmarkBase):
    def __init__(self, identifier, assembly, covariate_image_dir, similarity_metric, visual_degrees, number_of_trials, **kwargs):
        super(NeuralBenchmarkCovariate, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        self.covariate_image_dir = covariate_image_dir

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)

        # Find 'twin' image set whose model activations will serve as covariate
        brainio_dir = os.getenv('BRAINIO_HOME', os.path.join(os.path.expanduser('~'), '.brainio'))
        covariate_stimulus_set = pd.DataFrame(stimulus_set)
        covariate_stimulus_set = StimulusSet(covariate_stimulus_set)
        #covariate_stimulus_set = stimulus_set.copy(deep=True)
        covariate_stimulus_set.identifier = stimulus_set.identifier + '_' + self.covariate_image_dir
        covariate_stimulus_set.image_paths = {
            k: os.path.join(brainio_dir,
                            self.covariate_image_dir,
                            os.path.basename(v)) for k, v in stimulus_set.image_paths.items()}

        source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        source_assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier
        covariate_assembly = candidate.look_at(covariate_stimulus_set, number_of_trials=self._number_of_trials)
        covariate_assembly.attrs['stimulus_set_identifier'] = covariate_stimulus_set.identifier

        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            covariate_assembly = covariate_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, covariate_assembly, self._assembly)
        return explained_variance(raw_score, self.ceiling)

class CacheFeaturesCovariate(BenchmarkBase):
    def __init__(self, identifier, assembly, covariate_image_dir, similarity_metric, visual_degrees, number_of_trials, **kwargs):
        super(CacheFeaturesCovariate, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        self.covariate_image_dir = covariate_image_dir

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)

        # Find 'twin' image set whose model activations will serve as covariate
        brainio_dir = os.getenv('BRAINIO_HOME', os.path.join(os.path.expanduser('~'), '.brainio'))
        covariate_stimulus_set = pd.DataFrame(stimulus_set)
        covariate_stimulus_set = StimulusSet(covariate_stimulus_set)
        #covariate_stimulus_set = stimulus_set.copy(deep=True)
        covariate_stimulus_set.identifier = stimulus_set.identifier + '_' + self.covariate_image_dir
        covariate_stimulus_set.image_paths = {
            k: os.path.join(brainio_dir,
                            self.covariate_image_dir,
                            os.path.basename(v)) for k, v in stimulus_set.image_paths.items()}

        source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        source_assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier
        covariate_assembly = candidate.look_at(covariate_stimulus_set, number_of_trials=self._number_of_trials)
        covariate_assembly.attrs['stimulus_set_identifier'] = covariate_stimulus_set.identifier

        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            covariate_assembly = covariate_assembly.squeeze('time_bin')  # static case for these benchmarks
        #raw_score = self._similarity_metric(source_assembly, covariate_assembly, self._assembly)
        return 0

class NeuralBenchmarkCovariateGram(BenchmarkBase):
    def __init__(self, identifier, assembly, covariate_image_dir, similarity_metric, visual_degrees, number_of_trials, gram, **kwargs):
        super(NeuralBenchmarkCovariateGram, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        self.covariate_image_dir = covariate_image_dir
        self.gram = gram

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)


        # Find 'twin' image set whose model activations will serve as covariate
        brainio_dir = os.getenv('BRAINIO_HOME', os.path.join(os.path.expanduser('~'), '.brainio'))
        covariate_stimulus_set = pd.DataFrame(stimulus_set)
        covariate_stimulus_set = StimulusSet(covariate_stimulus_set)
        #covariate_stimulus_set = stimulus_set.copy(deep=True)
        covariate_stimulus_set.identifier = stimulus_set.identifier + '_' + self.covariate_image_dir
        covariate_stimulus_set.image_paths = {
            k: os.path.join(brainio_dir,
                            self.covariate_image_dir,
                            os.path.basename(v)) for k, v in stimulus_set.image_paths.items()}

        source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        source_assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier
        covariate_assembly = candidate.look_at(covariate_stimulus_set, number_of_trials=self._number_of_trials)
        covariate_assembly.attrs['stimulus_set_identifier'] = covariate_stimulus_set.identifier

        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            covariate_assembly = covariate_assembly.squeeze('time_bin')  # static case for these benchmarks

        if self.gram:
            model = covariate_assembly.model.values[0]
            layer = covariate_assembly.layer.values[0]
            fname = os.path.join(brainio_dir, self.covariate_image_dir, '_'.join([model, layer.replace('/', '_'), covariate_assembly.stimulus_set_identifier, 'gram.nc']))
            covariate_assembly = gram_on_all(covariate_assembly, fname = fname)
            source_assembly, covariate_assembly = source_assembly.sortby('image_id'), covariate_assembly.sortby('image_id')
            covariate_assembly = covariate_assembly.rename({'image_id': 'presentation'})
            covariate_assembly = covariate_assembly.assign_coords({'presentation': source_assembly.presentation.coords.to_index()})
            covariate_assembly = covariate_assembly.assign_coords({'neuroid': pd.MultiIndex.from_tuples(list(zip([model]*covariate_assembly.shape[0],
                                                                  [layer]*covariate_assembly.shape[0])), names=['model', 'layer'])})
            covariate_assembly.attrs['stimulus_set_identifier'] = covariate_stimulus_set.identifier

        raw_score = self._similarity_metric(source_assembly, covariate_assembly, self._assembly)
        return explained_variance(raw_score, self.ceiling)


class NeuralBenchmarkImageDir(BenchmarkBase):
    def __init__(self, identifier, assembly, image_dir, similarity_metric, visual_degrees, number_of_trials, **kwargs):
        super(NeuralBenchmarkImageDir, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assembly)
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        self.image_dir = image_dir

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set,
                                       target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)

        # Find 'twin' image set whose model activations we need
        brainio_dir = os.getenv('BRAINIO_HOME', os.path.join(os.path.expanduser('~'), '.brainio'))
        stimulus_set_from_dir = pd.DataFrame(stimulus_set)
        stimulus_set_from_dir = StimulusSet(stimulus_set_from_dir)
        # stimulus_set_from_dir = stimulus_set.copy(deep=True)
        stimulus_set_from_dir.identifier = stimulus_set.identifier + '_' + self.image_dir
        stimulus_set_from_dir.image_paths = {
            k: os.path.join(brainio_dir,
                            self.image_dir,
                            os.path.basename(v)) for k, v in stimulus_set.image_paths.items()}

        source_assembly = candidate.look_at(stimulus_set_from_dir, number_of_trials=self._number_of_trials)

        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return explained_variance(raw_score, self.ceiling)


class ToleranceCeiling(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, visual_degrees, number_of_trials, **kwargs):
        super(ToleranceCeiling, self).__init__(identifier=identifier, **kwargs)
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
        raw_score = self._similarity_metric(self._assembly)
        return explained_variance(raw_score)

def gram_on_all(assembly, fname):
    if os.path.isfile(fname):
        assembly = xr.open_dataarray(fname)
    else:
        assembly = assembly.T
        image_ids = assembly.image_id.values
        assembly = unflatten(assembly, channel_coord=None)
        assembly = assembly.reshape(list(assembly.shape[0:2]) + [-1])
        assembly = take_gram(assembly)
        assembly = assembly.T
        assembly = xr.DataArray(assembly, dims=['neuroid','image_id'], coords={'image_id':image_ids})
        assembly.to_netcdf(fname)
    return assembly


