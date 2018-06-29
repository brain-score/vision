import copy
import logging
import os

import caching
from caching import store

import brainscore
from brainscore.assemblies import merge_data_arrays
from brainscore.metrics.anatomy import ventral_stream, EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings
from brainscore.metrics.neural_fit import NeuralFit
from brainscore.metrics.rdm import RDMMetric
from brainscore.metrics.transformations import Transformations, CartesianProduct
from brainscore.utils import map_fields, combine_fields, fullname

caching.store.configure_storagedir(os.path.join(os.path.dirname(__file__), '..', 'output'))

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
    @store()
    def ceiling(self):
        return self._ceiling(self._target_assembly)

    def __repr__(self):
        return fullname(self)


class SplitBenchmark(Benchmark):
    def __init__(self, *args, target_splits, **kwargs):
        super(SplitBenchmark, self).__init__(*args, **kwargs)
        self._target_splits = target_splits
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
        scores = transformations(source_assembly, self._target_assembly, metric=self._metric)
        return scores

    def _ceil(self, score, ceiling):
        # since splits are the same when scoring and computing the ceiling, we can ceil both values and aggregate.
        ceiled_score = copy.deepcopy(score)
        for field in ['values', 'aggregation']:
            _score = getattr(ceiled_score, field)
            _ceiling = getattr(ceiling, field)
            for divider in self._target_dividers:
                selector = divider
                if field == 'aggregation':
                    selector = {**selector, **dict(aggregation='center')}
                _score.loc[selector] = _score.loc[selector] / _ceiling.loc[selector]
        return ceiled_score

    @property
    def ceiling(self):
        scores = []
        for i, divider in enumerate(self._target_dividers):
            self._logger.debug("Ceiling divider {}/{}: {}".format(i + 1, len(self._target_dividers), divider))
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


class DicarloMajaj2015(SplitBenchmark):
    def __init__(self):
        self._loader = DicarloMajaj2015Loader()
        assembly = self._loader(average_repetition=False)
        metric = metrics['neural_fit']()
        ceiling = ceilings['splitrep'](metric, average_repetition=self._loader.average_repetition)
        super(DicarloMajaj2015, self).__init__(assembly, metric, ceiling, target_splits=('region',))

    def _apply(self, source_assembly, source_splits=()):
        target_assembly_save = copy.deepcopy(self._target_assembly)
        self._target_assembly = self._loader.average_repetition(self._target_assembly)
        scores = super(DicarloMajaj2015, self)._apply(source_assembly, source_splits)
        self._target_assembly = target_assembly_save
        return scores


class GallantDavid2004(Benchmark):
    # work in progress
    def __init__(self):
        assembly = GallantDavid2004Loader()()
        metric = metrics['neural_fit'](regression='linear')
        ceiling = ceilings['cons']()
        super().__init__(assembly, metric, ceiling)


class FellemanVanEssen1991(Benchmark):
    # work in progress
    def __init__(self):
        self._target_assembly = ventral_stream
        self._metric = EdgeRatioMetric()


class AssemblyLoader(object):
    def __call__(self):
        raise NotImplementedError()


class DicarloMajaj2015Loader(AssemblyLoader):
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015')
        assembly.load()
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
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def average_repetition(self, assembly):
        return assembly.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')


class GallantDavid2004Loader(AssemblyLoader):
    def __call__(self):
        assembly = brainscore.get_assembly(name='gallant.David2004')
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly = assembly.stack(neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly


_assemblies = {
    'dicarlo.Majaj2015': DicarloMajaj2015Loader(),
    'gallant.David2004': GallantDavid2004Loader(),
}

_benchmarks = {
    'dicarlo.Majaj2015': DicarloMajaj2015,
    'gallant.David2004': GallantDavid2004,
    'Felleman1991': FellemanVanEssen1991,
}


def load(name):
    assert name in _benchmarks
    return _benchmarks[name]()


def load_assembly(name):
    return _assemblies[name]()


def build(assembly_name, metric_name, ceiling_name, target_splits=()):
    assembly = load_assembly(assembly_name)
    metric = metrics[metric_name]()
    ceiling = ceilings[ceiling_name]()
    return SplitBenchmark(assembly, metric, ceiling, target_splits=target_splits)
