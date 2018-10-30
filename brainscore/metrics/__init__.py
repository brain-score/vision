import warnings

from brainscore.assemblies import DataAssembly, merge_data_arrays


class Metric:
    def __call__(self, *args):
        raise NotImplementedError()


class Score(DataAssembly):
    RAW_VALUES_KEY = 'raw'

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"

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
