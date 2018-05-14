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

        self._ceiling = self.ceiling()

    def __call__(self, source_assembly, metric_kwargs=None):
        metric_kwargs = metric_kwargs or {}
        return self._metric(source_assembly, self._target_assembly, **metric_kwargs)

    def ceiling(self):
        return None
        # we need a way to pass in the cross-validation scheme here


def load(data_name, metric_name, metric_kwargs=None):
    metric_kwargs = metric_kwargs or {}
    if data_name == 'Felleman1991':
        data = ventral_stream
    else:
        data = mkgu.get_assembly(name=data_name)
        data.load()  # TODO: should load lazy
        if data_name == 'dicarlo.Majaj2015':
            data = data.sel(variation=6)  # TODO: remove variation selection once part of name
            data = data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim="presentation")
            data = data.squeeze("time_bin")
        if data_name == 'gallant.David2004':
            data = data.rename({'neuroid': 'neuroid_id'})
            # data['object_name'] = 'presentation', data['image_id']
            data = data.stack(neuroid=('neuroid_id',))
        if data_name.startswith('gallant'):
            metric_kwargs = {**dict(similarity_kwargs=dict(regression='linear')), **metric_kwargs}
        data = data.transpose('presentation', 'neuroid')
    # TODO: everything above this line needs work
    metric = metrics[metric_name](**metric_kwargs)
    return Benchmark(target_assembly=data, metric=metric)
