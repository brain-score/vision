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
        return build_score(DataAssembly([1]), center=DataAssembly([1]), error=DataAssembly([0]))


class SplitCeiling(Ceiling):
    def __init__(self, average_repetition,
                 repetition_train_size=0.5, repetition_test_size=0.5, repetition_dim='repetition'):
        self._average_repetition = average_repetition
        self._repetition_test_size = 1 / repetition_test_size
        self._rep_split = CrossValidation(train_size=repetition_train_size, test_size=repetition_test_size,
                                          dim=repetition_dim, stratification_coord=None)
        self._traintest_split = CrossValidation()
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly):
        repetition_values, repetition_splits = self._rep_split.build_splits(assembly)
        traintest_values, traintest_splits = self._traintest_split.build_splits(assembly)
        assert len(repetition_splits) == len(traintest_splits)
        # the following for loop is very similar to CrossValidation#__call__
        # but it didn't seem straight-forward how to merge the two.
        split_scores = []
        for split_iterator, ((half1_indices, half2_indices), (train_indices, test_indices)) \
                in enumerate(zip(repetition_splits, traintest_splits)):
            self._logger.debug("split {}/{}".format(split_iterator + 1, len(repetition_splits)))
            # split
            half1_values, half2_values = repetition_values[half1_indices], repetition_values[half2_indices]
            train_values, test_values = traintest_values[train_indices], traintest_values[test_indices]
            _subset = functools.partial(subset, dims_must_match=False)
            half1, half2 = _subset(assembly, half1_values), _subset(assembly, half2_values)
            half1, half2 = self._average_repetition(half1), self._average_repetition(half2)
            train_half1, test_half1 = _subset(half1, train_values), _subset(half1, test_values)
            train_half2, test_half2 = _subset(half2, train_values), _subset(half2, test_values)

            # compute
            split_score = self.score(train_half1, train_half2, test_half1, test_half2)
            split_score = spearman_brown_correction(split_score, n=self._repetition_test_size)

            # package
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = merge_data_arrays(split_scores)
        # aggregate
        center, error = self.aggregate(split_scores)
        return build_score(split_scores, center, error)

    def score(self, train_half1, train_half2, test_half1, test_half2):
        raise NotImplementedError()

    def aggregate(self, scores):
        return self._rep_split.aggregate(scores, scores)


class SplitRepCeiling(SplitCeiling):
    def __init__(self, metric, average_repetition, *args, **kwargs):
        self._metric = metric
        super(SplitRepCeiling, self).__init__(average_repetition=average_repetition, *args, **kwargs)

    def score(self, train_half1, train_half2, test_half1, test_half2):
        return self._metric(train_half1, train_half2, test_half1, test_half2)

    def aggregate(self, scores):
        scores = self._metric.aggregate(scores)
        return super(SplitRepCeiling, self).aggregate(scores)


class InternalConsistency(SplitCeiling):
    def __init__(self, *args, **kwargs):
        super(InternalConsistency, self).__init__(*args, **kwargs)

    def score(self, train_half1, train_half2, test_half1, test_half2):
        return ParametricMetric().compare_prediction(test_half1, test_half2)

    def aggregate(self, scores):
        scores = scores.median(dim='neuroid' if ('neuroid',) in scores.dims else 'neuroid_id')
        return super().aggregate(scores)


def spearman_brown_correction(correlation, n):
    return n * correlation / (1 + (n - 1) * correlation)


ceilings = {
    'splitrep': SplitRepCeiling,
    'cons': InternalConsistency,
    None: NoCeiling,
}
