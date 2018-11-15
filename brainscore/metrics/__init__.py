import warnings

from brainscore.assemblies import DataAssembly, merge_data_arrays


class Metric:
    def __call__(self, *args):
        raise NotImplementedError()


class Score(DataAssembly):
    RAW_VALUES_KEY = 'raw'

    def sel(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('sel', *args, **kwargs, _apply_raw=_apply_raw, _ignore_errors=True)

    def isel(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('isel', *args, **kwargs, _apply_raw=_apply_raw, _ignore_errors=True)

    def squeeze(self, *args, _apply_raw=True, **kwargs):
        return self._preserve_raw('squeeze', *args, **kwargs, _apply_raw=_apply_raw, _ignore_errors=True)

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

    def _preserve_raw(self, operation, *args, _apply_raw=False, _ignore_errors=False, **kwargs):
        result = getattr(super(Score, self), operation)(*args, **kwargs)
        if self.RAW_VALUES_KEY in self.attrs:
            raw = self.attrs[self.RAW_VALUES_KEY]
            if _apply_raw:
                try:
                    raw = getattr(raw, operation)(*args, **kwargs)
                except Exception as e:
                    if _ignore_errors:
                        # ignore errors with warning. most users will likely only want to access the main score
                        warnings.warn(f"{operation} on raw values failed: {repr(e)}")
                    else:
                        raise e
            result.attrs[self.RAW_VALUES_KEY] = raw
        return result

    def __setitem__(self, key, value):
        super(Score, self).__setitem__(key, value)
        if self.RAW_VALUES_KEY in self.attrs:
            try:
                self.attrs[self.RAW_VALUES_KEY].__setitem__(key, value)
            except Exception as e:
                warnings.warn(f"failed to set {key}={value} on raw values: " + (repr(e)))

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
