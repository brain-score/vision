import mkgu
from mkgu.metrics import Benchmark
from mkgu.metrics.neural_fit import NeuralFitMetric
from mkgu.metrics.rdm import RDMMetric

import xarray as xr

metrics = {
    'rdm': RDMMetric,
    'neural_fit': NeuralFitMetric
}


def load(data_name, metric_name):
    data_name = data_name.replace('dicarlo/hong2014', 'HvM')
    data = mkgu.get_assembly(name=data_name).sel(var='V6')
    data = data.loc[xr.ufuncs.logical_or(data["region"] == "V4", data["region"] == "IT")]
    data.load()  # TODO: should load lazy
    # TODO: everything above this line needs work
    metric = metrics[metric_name]()
    return Benchmark(assembly=data, metric=metric)
