from xarray import DataArray

from brainio_base.assemblies import merge_data_arrays, walk_coords
from brainscore.metrics.transformations import apply_aggregate


class TemporalRegressionAcrossTime:
    """
    Fits a regression with weights shared across the time bins.
    """

    def __init__(self, regression):
        self._regression = regression

    def fit(self, source, target):
        assert (source['time_bin'] == target['time_bin']).all()
        source = self._stack_timebins(source)
        target = self._stack_timebins(target)
        self._regression.fit(source, target)

    def predict(self, source):
        predictions = []
        for time_bin_start, time_bin_end in source['time_bin'].values:
            time_source = source.sel(time_bin=(time_bin_start, time_bin_end))
            prediction = self._regression.predict(time_source)
            prediction = prediction.expand_dims('time_bin_start').expand_dims('time_bin_end')
            prediction['time_bin_start'] = [time_bin_start]
            prediction['time_bin_end'] = [time_bin_end]
            prediction = prediction.stack(time_bin=['time_bin_start', 'time_bin_end'])
            predictions.append(prediction)
        return merge_data_arrays(predictions)

    def _stack_timebins(self, assembly):
        assembly_type = type(assembly)
        assembly = DataArray(assembly)  # xarray cannot deal with stacking MultiIndex (pydata/xarray#1554)
        assembly = assembly.reset_index(['presentation', 'time_bin'])
        assembly = assembly.rename({'presentation': '_presentation'})  # we'll call stacked timebins "presentation"
        assembly = assembly.stack(presentation=['_presentation', 'time_bin'])
        return assembly_type(assembly)


class TemporalCorrelationAcrossImages:
    """
    per time-bin, computes the correlation across images, then takes the mean across time-bins
    """

    def __init__(self, correlation):
        self._correlation = correlation

    def __call__(self, prediction, target):
        return cross_correlation(prediction, target, 'time_bin', self._correlation)


class TemporalCorrelationAcrossTime:
    """
    per image, computes the correlation across time-bins, then takes the mean across images
    """

    def __init__(self, correlation):
        self._correlation = correlation

    def __call__(self, prediction, target):
        return cross_correlation(prediction, target, 'presentation', self._correlation)


def cross_correlation(prediction, target, cross, correlation):
    assert (prediction[cross] == target[cross]).all()
    scores = []
    coords = [coord for coord, dims, values in walk_coords(target[cross])]
    for cross_value in target[cross].values:
        _prediction = prediction.sel(**{cross: cross_value})
        _target = target.sel(**{cross: cross_value})
        score = correlation(_prediction, _target)
        for coord, coord_value in zip(coords, cross_value):
            score = score.expand_dims(coord)
            score[coord] = [coord_value]
        score = score.stack(**{cross: coords})
        scores.append(score)
    score = merge_data_arrays(scores)
    score = apply_aggregate(lambda score: score.mean(cross), score)
    return score
