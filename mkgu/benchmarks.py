import functools

import xarray as xr

import mkgu
from mkgu.metrics import Benchmark
from mkgu.metrics.anatomy import ventral_stream, EdgeRatioMetric
from mkgu.metrics.neural_fit import NeuralFitMetric
from mkgu.metrics.rdm import RDMMetric

metrics = {
    'rdm': RDMMetric,
    'neural_fit': functools.partial(NeuralFitMetric, pca_components=1000),
    'edge_ratio': EdgeRatioMetric
}


def load(data_name, metric_name):
    if data_name == 'Felleman1991':
        data = ventral_stream
    else:
        data_name = data_name.replace('dicarlo/hong2014', 'HvM')
        data = mkgu.get_assembly(name=data_name).sel(var='V6')
        data = data.loc[xr.ufuncs.logical_or(data["region"] == "V4", data["region"] == "IT")]
        data.load()  # TODO: should load lazy
    # TODO: everything above this line needs work
    metric = metrics[metric_name]()
    return Benchmark(target_assembly=data, metric=metric)
