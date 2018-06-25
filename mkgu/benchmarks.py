from mkgu.assemblies import DataAssembly
from mkgu.metrics import Metric

import mkgu
from mkgu.metrics.anatomy import ventral_stream, EdgeRatioMetric
from mkgu.metrics.neural_fit import NeuralFit
from mkgu.metrics.rdm import RDMMetric
from mkgu.metrics.transformations import Transformations, CartesianProduct

metrics = {
    'rdm': RDMMetric,
    'neural_fit': NeuralFit,
    'edge_ratio': EdgeRatioMetric
}


class Benchmark(object):
    """a Benchmark represents the application of a Metric to a specific set of data.  """

    def __init__(self, metric: Metric, target_assembly: DataAssembly,
                 target_splits=CartesianProduct.Defaults.dividing_coords_target):
        self._metric = metric
        self._target_assembly = target_assembly
        self._target_splits = target_splits

    def __call__(self, source_assembly, source_splits=None):
        transformations = Transformations(cartesian_product_kwargs=dict(
            dividing_coord_names_source=source_splits, dividing_coord_names_target=self._target_splits))
        return transformations(source_assembly, self._target_assembly, metric=self._metric)

    @property
    def ceiling(self):
        return None
        # we need a way to pass in the cross-validation scheme here


def load(data_name, metric_name, metric_kwargs=None, target_assembly_splits=None):
    metric_kwargs = metric_kwargs or {}
    if data_name == 'Felleman1991':
        data = ventral_stream
    else:
        data = mkgu.get_assembly(name=data_name)
        data.load()
        if data_name == 'dicarlo.Majaj2015':
            err_neuroids = ['Tito_L_P_8_5', 'Tito_L_P_7_3', 'Tito_L_P_7_5', 'Tito_L_P_5_1', 'Tito_L_P_9_3',
                            'Tito_L_P_6_3', 'Tito_L_P_7_4', 'Tito_L_P_5_0', 'Tito_L_P_5_4', 'Tito_L_P_9_6',
                            'Tito_L_P_0_4', 'Tito_L_P_4_6', 'Tito_L_P_5_6', 'Tito_L_P_7_6', 'Tito_L_P_9_8',
                            'Tito_L_P_4_1', 'Tito_L_P_0_5', 'Tito_L_P_9_9', 'Tito_L_P_3_0', 'Tito_L_P_0_3',
                            'Tito_L_P_6_6', 'Tito_L_P_5_7', 'Tito_L_P_1_1', 'Tito_L_P_3_8', 'Tito_L_P_1_6',
                            'Tito_L_P_3_5', 'Tito_L_P_6_8', 'Tito_L_P_2_8', 'Tito_L_P_9_7', 'Tito_L_P_6_7',
                            'Tito_L_P_1_0', 'Tito_L_P_4_5', 'Tito_L_P_4_9', 'Tito_L_P_7_8', 'Tito_L_P_4_7',
                            'Tito_L_P_4_0', 'Tito_L_P_3_9', 'Tito_L_P_7_7', 'Tito_L_P_4_3', 'Tito_L_P_9_5']
            good_neuroids = [i for i, neuroid_id in enumerate(data['neuroid_id'].values)
                             if neuroid_id not in err_neuroids]
            data = data.isel(neuroid=good_neuroids)
            data = data.sel(variation=6)  # TODO: remove variation selection once part of name
            data = data.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim="presentation")
            data = data.squeeze("time_bin")
        if data_name == 'gallant.David2004':
            data = data.rename({'neuroid': 'neuroid_id'})
            data = data.stack(neuroid=('neuroid_id',))
        if data_name.startswith('gallant'):
            metric_kwargs = {**dict(regression='linear'), **metric_kwargs}
        data = data.transpose('presentation', 'neuroid')
        target_assembly_splits = target_assembly_splits or 'region'
    # TODO: everything above this line needs work
    metric = metrics[metric_name](**metric_kwargs)
    return Benchmark(target_assembly=data, metric=metric, target_splits=target_assembly_splits)
