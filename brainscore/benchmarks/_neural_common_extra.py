import numpy as np
import os
import pandas as pd

from brainio_base.assemblies import array_is_element, walk_coords
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from brainscore.benchmarks._neural_common import explained_variance, timebins_from_assembly
from brainio_base.stimuli import StimulusSet


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
        covariate_assembly = candidate.look_at(covariate_stimulus_set, number_of_trials=self._number_of_trials)

        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            covariate_assembly = covariate_assembly.squeeze('time_bin')  # static case for these benchmarks
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
