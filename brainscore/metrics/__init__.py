from brainscore.assemblies import merge_data_arrays
from brainscore.metrics.transformations import Alignment, Alignment, CartesianProduct, CrossValidation, \
    apply_transformations
from .utils import collect_coords, collect_dim_shapes, get_modified_coords, merge_dicts


class Metric(object):
    def __call__(self, *args):
        raise NotImplementedError()


class Score(object):
    def __init__(self, values, aggregation):
        self.values = values
        self.aggregation = aggregation

    def __repr__(self):
        return self.__class__.__name__ + "(" + ",".join(
            "{}={}".format(attr, val) for attr, val in self.__dict__.items()) + ")"

    def sel(self, *args, **kwargs):
        values = self.values.sel(*args, **kwargs)
        aggregation = self.aggregation.sel(*args, **kwargs)
        return Score(values, aggregation)

    def expand_dims(self, *args, **kwargs):
        values = self.values.expand_dims(*args, **kwargs)
        aggregation = self.aggregation.expand_dims(*args, **kwargs)
        return Score(values, aggregation)

    def __setitem__(self, key, value):
        self.values[key] = value
        self.aggregation[key] = value

    @classmethod
    def merge(cls, *scores):
        values = merge_data_arrays([score.values for score in scores])
        aggregation = merge_data_arrays([score.aggregation for score in scores])
        return Score(values, aggregation)


def build_score(values, center, error):
    # keep separate from Score class to keep constructor equal to fields (necessary for utils.py#combine_fields)
    center = center.expand_dims('aggregation')
    center['aggregation'] = ['center']
    error = error.expand_dims('aggregation')
    error['aggregation'] = ['error']
    aggregation = merge_data_arrays([center, error])
    return Score(values, aggregation)
