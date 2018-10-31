import warnings

from brainscore.assemblies import DataAssembly, merge_data_arrays


class Metric:
    def __call__(self, *args):
        raise NotImplementedError()


class Score(DataAssembly):
    RAW_VALUES_KEY = 'raw'

    def sel(self, *args, _select_raw=True, **kwargs):
        result = super().sel(*args, **kwargs)
        if _select_raw and self.RAW_VALUES_KEY in self.attrs:
            raw = self.attrs[self.RAW_VALUES_KEY]
            try:
                raw = raw.sel(*args, **kwargs)
            except Exception as e:
                # ignore errors with warning. most users will likely only want to access the main score
                warnings.warn("raw values not selected: " + repr(e))
            result.attrs[self.RAW_VALUES_KEY] = raw
        return result

    def expand_dims(self, *args, **kwargs):
        result = super(Score, self).expand_dims(*args, **kwargs)
        if self.RAW_VALUES_KEY in self.attrs:
            raw = self.attrs[self.RAW_VALUES_KEY].expand_dims(*args, **kwargs)
            result.attrs[self.RAW_VALUES_KEY] = raw
        return result

    def __setitem__(self, key, value):
        super(Score, self).__setitem__(key, value)
        if self.RAW_VALUES_KEY in self.attrs:
            self.attrs[self.RAW_VALUES_KEY].__setitem__(key, value)

    def _preserve_raw(self, operation, *args, raw_apply=False, **kwargs):
        result = getattr(super(Score, self), operation)(*args, **kwargs)
        if self.RAW_VALUES_KEY in self.attrs:
            raw = self.attrs[self.RAW_VALUES_KEY]
            if raw_apply:
                raw = getattr(raw, operation)(*args, **kwargs)
            result.attrs[self.RAW_VALUES_KEY] = raw
        return result

    def mean(self, *args, raw_apply=False, **kwargs):
        return self._preserve_raw('mean', *args, **kwargs, raw_apply=raw_apply)

    def sum(self, *args, raw_apply=False, **kwargs):
        return self._preserve_raw('sum', *args, **kwargs, raw_apply=raw_apply)

    def std(self, *args, raw_apply=False, **kwargs):
        return self._preserve_raw('std', *args, **kwargs, raw_apply=raw_apply)

    def min(self, *args, raw_apply=False, **kwargs):
        return self._preserve_raw('min', *args, **kwargs, raw_apply=raw_apply)

    @classmethod
    def merge(cls, *scores):
        """
        Merges the raw values in addition to the score assemblies.
        """
        result = merge_data_arrays(scores)
        raws = [score.attrs[cls.RAW_VALUES_KEY] for score in scores if cls.RAW_VALUES_KEY in score.attrs]
        if len(raws) > 0:
            raw = merge_data_arrays(raws)
            result.attrs[cls.RAW_VALUES_KEY] = raw
        return result
