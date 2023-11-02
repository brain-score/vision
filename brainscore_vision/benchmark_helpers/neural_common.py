import numpy as np

from brainio.assemblies import array_is_element, walk_coords
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
        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, self._assembly)
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
