"""
A :class:`~brainscore.metrics.Metric` is part of a :class:`~brainscore.benchmarks.Benchmark`
and scores how similar two sets of data are.
Typically these two sets are model and primate measurements, but metrics are agnostic of the data source
and can also be used to compare two primate measurements (e.g. for ceiling estimates).
"""

import warnings

import logging

from brainio_core.assemblies import DataAssembly, merge_data_arrays


class Metric:
    """
    Metric interface.
    A metric compares two sets of data and outputs a score of how well they match (1 = identical, 0 = no match).
    """

    def __call__(self, assembly1, assembly2):
        """
        Compare two assemblies on their similarity.
        These assemblies are typically neural or behavioral measurements, e.g. model and primate recordings.

        :param assembly1: the first assembly to compare against the second
        :param assembly2: the second assembly to compare against the first
        :return: a :class:`~brainscore.metrics.Score` denoting the match between the two assemblies
                (1 = identical, 0 = no match).
        """
        raise NotImplementedError()


_logger = logging.getLogger(__name__)  # cannot set directly on Score object


class Score(DataAssembly):
    """
    Scores are used as the outputs of metrics, benchmarks, and ceilings. They indicate similarity or goodness-of-fit
    of sets of data. The high-level score is typically an aggregate of many smaller scores, e.g. the median of neuroid
    correlations. To keep records of these smaller scores, a score can store "raw" scores in its attributes
    (`score.attrs['raw']`).
    """

    RAW_VALUES_KEY = 'raw'

    def sel(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('sel', *args, **kwargs, _apply_raw=_apply_raw)

    def isel(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('isel', *args, **kwargs, _apply_raw=_apply_raw)

    def squeeze(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('squeeze', *args, **kwargs, _apply_raw=_apply_raw)

    def expand_dims(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('expand_dims', *args, **kwargs, _apply_raw=_apply_raw)

    def mean(self, *args, _apply_raw=False, **kwargs):
        return self._preserve_raw('mean', *args, **kwargs, _apply_raw=_apply_raw)

    def sum(self, *args, _apply_raw=False, **kwargs):
        return self._preserve_raw('sum', *args, **kwargs, _apply_raw=_apply_raw)

    def std(self, *args, _apply_raw=False, **kwargs):
        return self._preserve_raw('std', *args, **kwargs, _apply_raw=_apply_raw)

    def min(self, *args, _apply_raw=False, **kwargs):
        return self._preserve_raw('min', *args, **kwargs, _apply_raw=_apply_raw)

    def _preserve_raw(self, operation, *args, _apply_raw=False, _ignore_errors=True, **kwargs):
        result = getattr(super(Score, self), operation)(*args, **kwargs)
        if self.RAW_VALUES_KEY in self.attrs:
            raw = self.attrs[self.RAW_VALUES_KEY]
            if _apply_raw:
                try:
                    raw = getattr(raw, operation)(*args, **kwargs)
                except Exception as e:
                    if _ignore_errors:
                        # ignore errors with warning. most users will likely only want to access the main score
                        _logger.debug(f"{operation} on raw values failed: {repr(e)}")
                    else:
                        raise e
            result.attrs[self.RAW_VALUES_KEY] = raw
        return result

    def __setitem__(self, key, value, _apply_raw=True):
        super(Score, self).__setitem__(key, value)
        if _apply_raw and self.RAW_VALUES_KEY in self.attrs:
            try:
                self.attrs[self.RAW_VALUES_KEY].__setitem__(key, value)
            except Exception as e:
                _logger.debug(f"failed to set {key}={value} on raw values: " + (repr(e)))

    @classmethod
    def merge(cls, *scores, ignore_exceptions=False):
        """
        Merges the raw values in addition to the score assemblies.
        """
        try:
            result = merge_data_arrays(scores)
            raws = [score.attrs[cls.RAW_VALUES_KEY] for score in scores if cls.RAW_VALUES_KEY in score.attrs]
            if len(raws) > 0:
                raw = Score.merge(*raws, ignore_exceptions=True)
                result.attrs[cls.RAW_VALUES_KEY] = raw
        except Exception as e:
            if ignore_exceptions:
                warnings.warn("failed to merge raw values: " + str(e))
                return None
            else:
                raise e
        return result
