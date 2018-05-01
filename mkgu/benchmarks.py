import xarray as xr

import mkgu
from mkgu.metrics.anatomy import ventral_stream, EdgeRatioMetric
from mkgu.metrics.neural_fit import NeuralFitMetric
from mkgu.metrics.rdm import RDMMetric

metrics = {
    'rdm': RDMMetric,
    'neural_fit': NeuralFitMetric,
    'edge_ratio': EdgeRatioMetric
}


class Benchmark(object):
    """a Benchmark represents the application of a Metric to a specific set of data.  """

    def __init__(self, metric, target_assembly):
        """
        :param Metric metric:
        :param target_assembly:
        """
        self._metric = metric
        self._target_assembly = target_assembly

    def __call__(self, source_assembly):
        return self._metric(self._target_assembly, source_assembly)


def load(data_name, metric_name):
    if data_name == 'Felleman1991':
        data = ventral_stream
    else:
        data = mkgu.get_assembly(name=data_name).sel(variation=6)  # TODO: remove variation selection once part of name
        data = data.loc[xr.ufuncs.logical_or(data["region"] == "V4", data["region"] == "IT")]
        data.load()  # TODO: should load lazy
        data = data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim="presentation")
        data = data.squeeze("time_bin")
        data = data.transpose('presentation', 'neuroid')
    # TODO: everything above this line needs work
    metric = metrics[metric_name]()
    return Benchmark(target_assembly=data, metric=metric)
