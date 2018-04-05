import functools

import xarray as xr

import mkgu
from mkgu.metrics import Benchmark
from mkgu.metrics.neural_fit import NeuralFitMetric
from mkgu.metrics.rdm import RDMMetric

metrics = {
    'rdm': RDMMetric,
    'neural_fit': functools.partial(NeuralFitMetric, pca_components=1000)
}


def load(data_name, metric_name):
    data_name = data_name.replace('dicarlo/hong2014', 'HvM')
    data = mkgu.get_assembly(name=data_name).sel(var='V6')
    data = data.loc[xr.ufuncs.logical_or(data["region"] == "V4", data["region"] == "IT")]
    data.load()  # TODO: should load lazy
    data = data.multi_groupby(['obj', 'id']).mean(dim='presentation').squeeze('time_bin')
    data = data.T
    # TODO: everything above this line needs work
    metric = metrics[metric_name]()
    return Benchmark(target_assembly=data, metric=metric)
