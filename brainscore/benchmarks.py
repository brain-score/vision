import copy
import logging
import os

import numpy as np

import brainscore
import caching
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element
from brainscore.metrics import NonparametricWrapper
from brainscore.metrics.anatomy import ventral_stream, EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings
from brainscore.metrics.neural_fit import PlsFit, LinearFit
from brainscore.metrics.rdm import RDMMetric
from brainscore.metrics.transformations import Transformations, CartesianProduct
from brainscore.utils import map_fields, combine_fields, fullname, recursive_dict_merge
from caching import store

caching.store.configure_storagedir(os.path.join(os.path.dirname(__file__), '..', 'output'))


class Benchmark(object):
    def __init__(self, target_assembly, metric):
        self._target_assembly = target_assembly
        self.stimulus_set_name = target_assembly.attrs['stimulus_set_name']
        self._metric = metric
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, identifier=None):
        if identifier is None and source_assembly.name is None:
            raise ValueError("must provide either identifier or source_assembly.name")
        identifier = identifier or source_assembly.name
        scores = self._call(source_assembly, identifier=identifier)
        return scores

    @store(identifier_ignore=['source_assembly'])
    def _call(self, source_assembly, identifier):
        return self._apply(source_assembly)

    def _apply(self, source_assembly):
        return self._metric(source_assembly, self._target_assembly)

    def __repr__(self):
        return fullname(self)


class BrainScore(Benchmark):
    def __init__(self):
        super(BrainScore, self).__init__(None, None)
        self._v4_it_benchmark = DicarloMajaj2015()
        self._kwargs = None

    def __call__(self, source_assembly, identifier=None, **kwargs):
        self._kwargs = kwargs
        scores = super(BrainScore, self).__call__(source_assembly, identifier=identifier)
        self._kwargs = None
        return scores

    def _apply(self, source_assembly):
        v4_it_score = self._v4_it_benchmark(source_assembly, **self._kwargs)
        v4_it_score = v4_it_score.aggregation
        aggregation_dims = ['aggregation', 'region']
        assert all(dim in v4_it_score.dims for dim in aggregation_dims)
        reduce_dims = [dim for dim in v4_it_score.dims if dim not in aggregation_dims]
        v4_it_score = v4_it_score.max(reduce_dims)
        np.testing.assert_array_equal(v4_it_score.dims, aggregation_dims)
        v4_it_score = v4_it_score.sel(aggregation='center')
        brain_score = np.mean(v4_it_score)
        # TODO: behavior
        return brain_score


class CeiledBenchmark(Benchmark):
    def __init__(self, target_assembly, metric, ceiling):
        super(CeiledBenchmark, self).__init__(target_assembly=target_assembly, metric=metric)
        self._ceiling = ceiling
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, identifier=None, return_unceiled=False):
        scores = super(CeiledBenchmark, self).__call__(source_assembly, identifier=identifier)
        ceiled_scores = self._ceil(scores, self.ceiling)
        if return_unceiled:
            return ceiled_scores, scores
        return ceiled_scores

    def _ceil(self, scores, ceiling):
        return scores / ceiling

    @property
    @store()
    def ceiling(self):
        return self._apply_ceiling(self._target_assembly)

    def _apply_ceiling(self, assembly):
        return self._ceiling(assembly)


class SplitBenchmark(CeiledBenchmark):
    def __init__(self, *args, target_splits=(), target_splits_kwargs=None, **kwargs):
        super(SplitBenchmark, self).__init__(*args, **kwargs)
        self._target_splits = target_splits
        target_splits_kwargs = target_splits_kwargs or {}
        cartesian_product = CartesianProduct(**target_splits_kwargs)
        self._target_dividers = cartesian_product.dividers(self._target_assembly, self._target_splits)
        self._transformations = None

    def __call__(self, source_assembly, identifier=None, transformation_kwargs=None, return_unceiled=False):
        transformation_kwargs = transformation_kwargs or {}
        target_split_kwargs = dict(cartesian_product_kwargs=dict(dividing_coord_names_target=self._target_splits))
        transformation_kwargs = recursive_dict_merge(target_split_kwargs, transformation_kwargs)
        self._transformations = Transformations(**transformation_kwargs)
        scores = super(SplitBenchmark, self).__call__(source_assembly, identifier=identifier,
                                                      return_unceiled=return_unceiled)
        self._transformations = None
        return scores

    def _apply(self, source_assembly):
        scores = self._transformations(source_assembly, self._target_assembly, metric=self._metric)
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
                # handle same-name regions in source and target
                score_selector = self._check_selector_coord_names(selector, _score)
                _score.loc[score_selector] = _score.loc[score_selector] / _ceiling.loc[selector]
        return ceiled_score

    def _check_selector_coord_names(self, selector, assembly):
        selector = copy.deepcopy(selector)
        score_selector_items = set(selector.items())
        for div_name, div_value in score_selector_items:
            if not hasattr(assembly, div_name):
                div_src = CartesianProduct.rename_duplicate_coord(div_name, CartesianProduct.CoordType.SOURCE)
                div_tgt = CartesianProduct.rename_duplicate_coord(div_name, CartesianProduct.CoordType.TARGET)
                assert hasattr(assembly, div_src) and hasattr(assembly, div_tgt)
                del selector[div_name]
                selector[div_tgt] = div_value  # we only need target for the ceiling correction
        return selector

    def _apply_ceiling(self, assembly):
        scores = []
        for i, divider in enumerate(self._target_dividers):
            self._logger.debug("Ceiling divider {}/{}: {}".format(i + 1, len(self._target_dividers), divider))
            div_assembly = assembly.multisel(**divider)
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
        metric = metrics['pls_fit']()
        ceiling = ceilings['splitrep'](metric, average_repetition=self._loader.average_repetition)
        super(DicarloMajaj2015, self).__init__(assembly, metric, ceiling, target_splits=('region',))

    def _apply(self, source_assembly):
        target_assembly_save = copy.deepcopy(self._target_assembly)
        self._target_assembly = self._loader.average_repetition(self._target_assembly)
        scores = super(DicarloMajaj2015, self)._apply(source_assembly)
        self._target_assembly = target_assembly_save
        return scores


class ToliasCadena2017(SplitBenchmark):
    def __init__(self):
        self._loader = ToliasCadena2017Loader()
        assembly = self._loader(average_repetition=False)
        metric = metrics['pls_fit']()
        ceiling = ceilings['splitrep'](metric, repetition_dim='repetition_id',
                                       average_repetition=self._loader.average_repetition)
        super(ToliasCadena2017, self).__init__(assembly, metric, ceiling)

    def _apply(self, source_assembly):
        target_assembly_save = copy.deepcopy(self._target_assembly)
        self._target_assembly = self._loader.average_repetition(self._target_assembly)
        scores = super(ToliasCadena2017, self)._apply(source_assembly)
        self._target_assembly = target_assembly_save
        return scores


class GallantDavid2004(CeiledBenchmark):
    # work in progress
    def __init__(self):
        assembly = GallantDavid2004Loader()()
        metric = metrics['pls_fit'](regression='linear')
        ceiling = ceilings['cons']()
        super().__init__(assembly, metric, ceiling)


class FellemanVanEssen1991(CeiledBenchmark):
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
        attrs = copy.deepcopy(assembly.attrs)
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
        assembly.attrs = attrs
        return assembly

    def average_repetition(self, assembly):
        return assembly.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')


class ToliasCadena2017Loader(AssemblyLoader):
    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        attrs = copy.deepcopy(assembly.attrs)
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly['region'] = 'neuroid_id', ['V1'] * len(assembly['neuroid_id'])
        assembly = assembly.stack(neuroid=['neuroid_id'])
        assembly = assembly.squeeze("time_bin")
        # TODO: instead of discarding the entire image, see if we can be smarter in the metrics
        assembly = assembly.dropna('presentation')  # discard any images with NaNs (~56%)
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        assembly.attrs = attrs
        return assembly

    def average_repetition(self, assembly):
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        return assembly.multi_groupby(presentation_coords).mean(dim='presentation')


class GallantDavid2004Loader(AssemblyLoader):
    def __call__(self):
        assembly = brainscore.get_assembly(name='gallant.David2004')
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly = assembly.stack(neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly


metrics = {
    'rdm': lambda *args, **kwargs: NonparametricWrapper(RDMMetric(*args, **kwargs)),
    'linear_fit': LinearFit,
    'pls_fit': PlsFit,
    'edge_ratio': EdgeRatioMetric
}

assembly_loaders = {
    'dicarlo.Majaj2015': DicarloMajaj2015Loader(),
    'gallant.David2004': GallantDavid2004Loader(),
    'tolias.Cadena2017': ToliasCadena2017Loader(),
}

_benchmarks = {
    'brain-score': BrainScore,
    'dicarlo.Majaj2015': DicarloMajaj2015,
    'gallant.David2004': GallantDavid2004,
    'Felleman1991': FellemanVanEssen1991,
    'tolias.Cadena2017': ToliasCadena2017,
}


def load(name):
    if name not in _benchmarks:
        raise ValueError("Unknown benchmark '{}' - must choose from {}".format(name, list(_benchmarks.keys())))
    return _benchmarks[name]()


def load_assembly(name: str):
    """
    Loads the assembly using an AssemblyLoader.
    The AssemblyLoader might further refine the raw assembly provided by brainscore.get_assembly.
    :param name: the name of the assembly loader
    :return: the loaded assembly
    """
    return assembly_loaders[name]()


def build(assembly_name, metric_name, ceiling_name=None, target_splits=()):
    assembly = load_assembly(assembly_name)
    metric = metrics[metric_name]()
    ceiling = ceilings[ceiling_name]()
    return SplitBenchmark(assembly, metric, ceiling, target_splits=target_splits)


if __name__ == '__main__':
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    benchmark = load('brain-score')
    source = load_assembly('dicarlo.Majaj2015')
    score = benchmark(source, transformation_kwargs=dict(
        cartesian_product_kwargs=dict(dividing_coord_names_source=['region'])))
    assert score == 1
