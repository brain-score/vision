import copy
import logging

import mkgu
from mkgu.assemblies import merge_data_arrays
from mkgu.metrics.anatomy import ventral_stream, EdgeRatioMetric
from mkgu.metrics.ceiling import ceilings
from mkgu.metrics.neural_fit import NeuralFit
from mkgu.metrics.rdm import RDMMetric
from mkgu.metrics.transformations import Transformations, CartesianProduct
from mkgu.utils import map_fields, combine_fields, fullname

metrics = {
    'rdm': RDMMetric,
    'neural_fit': NeuralFit,
    'edge_ratio': EdgeRatioMetric
}


class Benchmark(object):
    def __init__(self, target_assembly, metric, ceiling):
        self._target_assembly = target_assembly
        self._metric = metric
        self._ceiling = ceiling
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, return_unceiled=False):
        scores = self._apply(source_assembly)
        ceiled_scores = self._ceil(scores, self.ceiling)
        if return_unceiled:
            return ceiled_scores, scores
        return ceiled_scores

    def _apply(self, source_assembly):
        return self._metric(source_assembly, self._target_assembly)

    def _ceil(self, scores, ceiling):
        return scores / ceiling

    @property
    def ceiling(self):
        return self._ceiling(self._target_assembly)

    def _load_assembly(self, name):
        assembly = mkgu.get_assembly(name=name)
        assembly.load()
        return assembly

    def __repr__(self):
        return fullname(self)


class DicarloMajaj2015(Benchmark):
    def __init__(self):
        assembly = self._load_assembly('dicarlo.Majaj2015')
        metric = metrics['neural_fit']()
        ceiling = ceilings['splitrep'](metric, average_repetition=self.average_repetition)
        super(DicarloMajaj2015, self).__init__(assembly, metric, ceiling)

        self._target_splits = ('region',)
        cartesian_product = CartesianProduct()
        self._target_dividers = cartesian_product.dividers(self._target_assembly, self._target_splits)

    def __call__(self, source_assembly, source_splits=(), return_unceiled=False):
        scores = self._apply(source_assembly, source_splits=source_splits)
        ceiled_scores = self._ceil(scores, self.ceiling)
        if return_unceiled:
            return ceiled_scores, scores
        return ceiled_scores

    def _apply(self, source_assembly, source_splits=()):
        transformations = Transformations(cartesian_product_kwargs=dict(
            dividing_coord_names_source=source_splits, dividing_coord_names_target=self._target_splits))
        target_assembly = self.average_repetition(self._target_assembly)
        scores = transformations(source_assembly, target_assembly, metric=self._metric)
        return scores

    def _ceil(self, score, ceiling):
        # since splits are independent between score and ceiling, we can only really look at the aggregate.
        # Ideally, we would want to ceil every single split but without equal seeding, we cannot do that currently.
        ceiled_score = copy.deepcopy(score)
        ceiled_score.values = None
        for divider in self._target_dividers:
            _score = ceiled_score.aggregation
            _ceiling = ceiling.aggregation
            selector = {**divider, **dict(aggregation='center')}
            _score.loc[selector] = _score.loc[selector] / _ceiling.loc[selector]
        return ceiled_score

    def average_repetition(self, assembly):
        return assembly.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')

    def _load_assembly(self, name):
        assembly = super()._load_assembly(name)
        err_neuroids = ['Tito_L_P_8_5', 'Tito_L_P_7_3', 'Tito_L_P_7_5', 'Tito_L_P_5_1', 'Tito_L_P_9_3',
                        'Tito_L_P_6_3', 'Tito_L_P_7_4', 'Tito_L_P_5_0', 'Tito_L_P_5_4', 'Tito_L_P_9_6',
                        'Tito_L_P_0_4', 'Tito_L_P_4_6', 'Tito_L_P_5_6', 'Tito_L_P_7_6', 'Tito_L_P_9_8',
                        'Tito_L_P_4_1', 'Tito_L_P_0_5', 'Tito_L_P_9_9', 'Tito_L_P_3_0', 'Tito_L_P_0_3',
                        'Tito_L_P_6_6', 'Tito_L_P_5_7', 'Tito_L_P_1_1', 'Tito_L_P_3_8', 'Tito_L_P_1_6',
                        'Tito_L_P_3_5', 'Tito_L_P_6_8', 'Tito_L_P_2_8', 'Tito_L_P_9_7', 'Tito_L_P_6_7',
                        'Tito_L_P_1_0', 'Tito_L_P_4_5', 'Tito_L_P_4_9', 'Tito_L_P_7_8', 'Tito_L_P_4_7',
                        'Tito_L_P_4_0', 'Tito_L_P_3_9', 'Tito_L_P_7_7', 'Tito_L_P_4_3', 'Tito_L_P_9_5']
        good_neuroids = [i for i, neuroid_id in enumerate(assembly['neuroid_id'].values)
                         if neuroid_id not in err_neuroids]
        assembly = assembly.isel(neuroid=good_neuroids)
        assembly = assembly.sel(variation=6)  # TODO: remove variation selection once part of name
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly

    @property
    def ceiling(self):
        scores = []
        for i, divider in enumerate(self._target_dividers):
            self._logger.debug("Ceiling divider {}/{}: {}".format(i, len(self._target_dividers), divider))
            div_assembly = self._target_assembly.multisel(**divider)
            score = self._ceiling(div_assembly)

            def insert_dividers(data):
                for coord_name, coord_value in divider.items():
                    data = data.expand_dims(coord_name)
                    data[coord_name] = [coord_value]
                return data

            map_fields(score, insert_dividers)
            scores.append(score)

        scores = combine_fields(scores, merge_data_arrays)
        return scores


class DicarloMajaj2015RDM(DicarloMajaj2015):
    def __init__(self):
        super(DicarloMajaj2015, self).__init__(assembly_name='dicarlo.Majaj2015', metric_name='rdm',
                                               ceiling_name='splitrep', target_splits=('region',))


class GallantDavid2004(Benchmark):
    def _load_assembly(self, name):
        assembly = super()._load_assembly(name)
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly = assembly.stack(neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly

    def _load_metric(self, name):
        return metrics[name](regression='linear')


class FellemanVanEssen1991(Benchmark):
    def __init__(self):
        self._target_assembly = ventral_stream
        self._metric = EdgeRatioMetric()


_benchmarks = {
    'dicarlo.Majaj2015': DicarloMajaj2015,
    'dicarlo.Majaj2015-rdm': DicarloMajaj2015RDM,
    'gallant.David2004': GallantDavid2004,
    'Felleman1991': FellemanVanEssen1991,
}


def load(name):
    assert name in _benchmarks
    return _benchmarks[name]()
