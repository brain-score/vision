import functools
import logging

from mkgu.assemblies import merge_data_arrays, DataAssembly
from mkgu.metrics import ParametricMetric, build_score
from mkgu.metrics.transformations import CrossValidation, subset
from mkgu.utils import fullname


class Ceiling(object):
    def __call__(self, assembly):
        raise NotImplementedError()


class NoCeiling(Ceiling):
    def __call__(self, assembly):
        return build_score(DataAssembly(1), center=DataAssembly(1), error=DataAssembly(0))


class SplitCeiling(Ceiling):
    def __init__(self, n_splits=CrossValidation.Defaults.splits):
        self._n_splits = n_splits
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly):
        split_scores = []
        # the following for loop is very similar to CrossValidation#__call__
        # but it didn't seem straight-forward how to merge the two.
        for split_iterator in range(self._n_splits):
            self._logger.debug("split {}/{}".format(split_iterator + 1, self._n_splits))
            split_score = self._compute_split(assembly)
            # package
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)
        split_scores = merge_data_arrays(split_scores)
        # aggregate
        center, error = self.aggregate(split_scores)
        return build_score(split_scores, center, error)

    def _compute_split(self, assembly):
        raise NotImplementedError()

    def aggregate(self, scores):
        raise NotImplementedError()


class SplitNoCeiling(SplitCeiling):
    def _compute_split(self, assembly):
        return DataAssembly(1)

    def aggregate(self, scores):
        return DataAssembly(1), DataAssembly(0)


class SplitRepCeiling(SplitCeiling):
    def __init__(self, average_repetition, n_splits=CrossValidation.Defaults.splits,
                 repetition_train_size=0.5, repetition_test_size=0.5, repetition_dim='repetition'):
        super(SplitRepCeiling, self).__init__(n_splits=n_splits)
        self._average_repetition = average_repetition
        self._repetition_test_size = 1 / repetition_test_size
        self._rep_split = CrossValidation(train_size=repetition_train_size, test_size=repetition_test_size,
                                          dim=repetition_dim, stratification_coord=None, splits=n_splits)
        self._traintest_split = CrossValidation(splits=n_splits)
        self._logger = logging.getLogger(fullname(self))
        self._split_generator = None
        self._repetition_values, self._traintest_values = None, None

    def __call__(self, assembly):
        repetition_values, repetition_splits = self._rep_split.build_splits(assembly)
        traintest_values, traintest_splits = self._traintest_split.build_splits(assembly)
        assert len(repetition_splits) == len(traintest_splits)
        self._repetition_values, self._traintest_values = repetition_values, traintest_values
        self._split_generator = zip(repetition_splits, traintest_splits)
        result = super(SplitRepCeiling, self).__call__(assembly)
        self._split_generator = None  # reset
        self._repetition_values, self._traintest_values = None, None
        return result

    def _compute_split(self, assembly):
        ((half1_indices, half2_indices), (train_indices, test_indices)) = next(self._split_generator)
        half1_values, half2_values = self._repetition_values[half1_indices], self._repetition_values[half2_indices]
        train_values, test_values = self._traintest_values[train_indices], self._traintest_values[test_indices]
        _subset = functools.partial(subset, dims_must_match=False)
        half1, half2 = _subset(assembly, half1_values), _subset(assembly, half2_values)
        half1, half2 = self._average_repetition(half1), self._average_repetition(half2)
        train_half1, test_half1 = _subset(half1, train_values), _subset(half1, test_values)
        train_half2, test_half2 = _subset(half2, train_values), _subset(half2, test_values)

        # compute
        split_score = self.score(train_half1, train_half2, test_half1, test_half2)
        split_score = spearman_brown_correction(split_score, n=self._repetition_test_size)
        return split_score

    def score(self, train_half1, train_half2, test_half1, test_half2):
        raise NotImplementedError()

    def aggregate(self, scores):
        return self._rep_split.aggregate(scores, scores)


class SplitRepMetricCeiling(SplitRepCeiling):
    def __init__(self, metric, *args, **kwargs):
        self._metric = metric
        super().__init__(*args, **kwargs)

    def score(self, train_half1, train_half2, test_half1, test_half2):
        return self._metric(train_half1, train_half2, test_half1, test_half2)

    def aggregate(self, scores):
        scores = self._metric.aggregate(scores)
        return super().aggregate(scores)


class InternalConsistency(SplitRepCeiling):
    def score(self, train_half1, train_half2, test_half1, test_half2):
        return ParametricMetric().compare_prediction(test_half1, test_half2)

    def aggregate(self, scores):
        scores = scores.median(dim='neuroid' if ('neuroid',) in scores.dims else 'neuroid_id')
        return super().aggregate(scores)


def spearman_brown_correction(correlation, n):
    return n * correlation / (1 + (n - 1) * correlation)


ceilings = {
    'splitrep': SplitRepMetricCeiling,
    'cons': InternalConsistency,
    None: NoCeiling,
}
