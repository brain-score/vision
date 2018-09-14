import copy
import logging

import numpy as np
from result_caching import cache, store

import brainscore
from brainscore.assemblies import merge_data_arrays, walk_coords, array_is_element, DataAssembly
from brainscore.metrics import NonparametricWrapper, Score
from brainscore.metrics.anatomy import ventral_stream, EdgeRatioMetric
from brainscore.metrics.ceiling import ceilings
from brainscore.metrics.neural_predictivity import PlsPredictivity, LinearPredictivity
from brainscore.metrics.rdm import RDMMetric
from brainscore.metrics.transformations import Transformations, CartesianProduct
from brainscore.utils import map_fields, combine_fields, fullname, recursive_dict_merge


class Benchmark(object):
    def __init__(self, name, target_assembly, metric):
        self.name = name
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


class BrainScore(object):
    # Brain-Score is a Benchmark too, but due to its compositionality
    # we deem it too different from the Benchmark base class.
    def __init__(self):
        self.name = 'brain-score'
        self._v4_it_benchmark = DicarloMajaj2015()
        # hacky but works for now: copy stimulus set from child benchmark.
        # this will not scale to multiple child benchmarks.
        self.stimulus_set_name = self._v4_it_benchmark.stimulus_set_name
        self._kwargs = None

    def __call__(self, source_assembly, identifier=None, **kwargs):
        v4_it_score = self._v4_it_benchmark(source_assembly, **kwargs)
        v4_it_score = v4_it_score.aggregation
        aggregation_dims = ['aggregation', 'region']
        assert all(dim in v4_it_score.dims for dim in aggregation_dims)
        reduce_dims = [dim for dim in v4_it_score.dims if dim not in aggregation_dims]
        aggregate_v4_it_score = v4_it_score.max(reduce_dims)
        np.testing.assert_array_equal(aggregate_v4_it_score.dims, aggregation_dims)
        aggregate_v4_it_score = aggregate_v4_it_score.sel(aggregation='center')
        brain_score = np.mean(aggregate_v4_it_score).values
        # TODO: behavior
        values = v4_it_score.expand_dims('benchmark')
        values['benchmark'] = [self._v4_it_benchmark.name]
        aggregation = DataAssembly([brain_score], coords={'aggregation': ['center']}, dims=['aggregation'])
        score = Score(aggregation=aggregation, values=values)
        return score


class CeiledBenchmark(Benchmark):
    def __init__(self, *args, ceiling, **kwargs):
        super(CeiledBenchmark, self).__init__(*args, **kwargs)
        self._ceiling = ceiling
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_assembly, identifier=None, return_ceiled=False):
        scores = super(CeiledBenchmark, self).__call__(source_assembly, identifier=identifier)
        ceiled_scores = self._ceil(scores, self.ceiling)
        if return_ceiled:
            return ceiled_scores, scores
        return scores

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

    def __call__(self, source_assembly, identifier=None, transformation_kwargs=None, return_ceiled=False):
        transformation_kwargs = transformation_kwargs or {}
        target_split_kwargs = dict(cartesian_product_kwargs=dict(dividing_coord_names_target=self._target_splits))
        transformation_kwargs = recursive_dict_merge(target_split_kwargs, transformation_kwargs)
        self._transformations = Transformations(**transformation_kwargs)
        scores = super(SplitBenchmark, self).__call__(source_assembly, identifier=identifier,
                                                      return_ceiled=return_ceiled)
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
                    data[coord_name] = [coord_value]  # TODO: multi-index dims
                return data

            map_fields(score, insert_dividers)
            scores.append(score)

        scores = combine_fields(scores, merge_data_arrays)
        return scores


class DicarloMajaj2015(SplitBenchmark):
    def __init__(self):
        self._loader = DicarloMajaj2015Loader()
        assembly = self._loader(average_repetition=False)
        metric = metrics['pls_predictivity']()
        ceiling = ceilings['splitrep'](metric, average_repetition=self._loader.average_repetition)
        super(DicarloMajaj2015, self).__init__(name='dicarlo.Majaj2015', target_assembly=assembly,
                                               metric=metric, ceiling=ceiling, target_splits=('region',))

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
        metric = metrics['pls_predictivity']()
        ceiling = ceilings['splitrep'](metric, repetition_dim='repetition_id',
                                       average_repetition=self._loader.average_repetition)
        super(ToliasCadena2017, self).__init__(name='tolias.Cadena2017',
                                               target_assembly=assembly, metric=metric, ceiling=ceiling)

    def _apply(self, source_assembly):
        target_assembly_save = copy.deepcopy(self._target_assembly)
        self._target_assembly = self._loader.average_repetition(self._target_assembly)
        scores = super(ToliasCadena2017, self)._apply(source_assembly)
        self._target_assembly = target_assembly_save
        return scores


class DicarloMajaj2015EarlyLate(DicarloMajaj2015):
    def __init__(self):
        self._loader = DicarloMajaj2015EarlyLateLoader()
        assembly = self._loader(average_repetition=False)
        metric = metrics['pls_predictivit']()
        ceiling = ceilings['splitrep'](metric, average_repetition=self._loader.average_repetition)
        SplitBenchmark.__init__(self, assembly, metric, ceiling,
                                target_splits=('region', 'time_bin_start', 'time_bin_end'))


class GallantDavid2004(CeiledBenchmark):
    # work in progress
    def __init__(self):
        assembly = GallantDavid2004Loader()()
        metric = metrics['pls_predictivity'](regression='linear')
        ceiling = ceilings['cons']()
        super().__init__(name='gallant.David2004', target_assembly=assembly, metric=metric, ceiling=ceiling)


class FellemanVanEssen1991(CeiledBenchmark):
    # work in progress
    def __init__(self):
        self._target_assembly = ventral_stream
        self._metric = EdgeRatioMetric()


class AssemblyLoader(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        raise NotImplementedError()


class DicarloMajaj2015Loader(AssemblyLoader):
    def __init__(self, name='dicarlo.Majaj2015'):
        super(DicarloMajaj2015Loader, self).__init__(name=name)

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = self._filter_erroneous_neuroids(assembly)
        assembly = assembly.sel(variation=6)  # TODO: remove variation selection once part of name
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def _filter_erroneous_neuroids(self, assembly):
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
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        assembly = assembly.multi_groupby(['category_name', 'object_name', 'image_id']).mean(dim='presentation')
        assembly.attrs = attrs
        return assembly


class DicarloMajaj2015TemporalLoader(DicarloMajaj2015Loader):
    def __init__(self, name='dicarlo.Majaj2015.temporal'):
        super(DicarloMajaj2015TemporalLoader, self).__init__(name=name)

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='dicarlo.Majaj2015.temporal')
        assembly = self._filter_erroneous_neuroids(assembly)
        assembly = assembly.sel(variation=6)  # TODO: remove variation selection once part of name
        assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly


class DicarloMajaj2015EarlyLateLoader(DicarloMajaj2015TemporalLoader):
    def __init__(self):
        super(DicarloMajaj2015EarlyLateLoader, self).__init__(name='dicarlo.Majaj2015.earlylate')

    @store()
    def __call__(self, average_repetition=True):
        assembly = super().__call__(average_repetition=average_repetition)

        # the principled way here would be to compute the internal consistency per neuron
        # and only use the time bins where the neuron's internal consistency is e.g. >0.2.
        # Note that this would have to be done per neuron
        # as a neuron in V4 will exhibit different time bins from one in IT.
        # We cut off at 200 ms because that's when the next stimulus is shown.
        def sel_time_bin(time_bin_start, time_bin_end):
            selection = assembly.sel(time_bin_start=time_bin_start, time_bin_end=time_bin_end)
            del selection['time_bin']
            selection = selection.expand_dims('time_bin_start').expand_dims('time_bin_end')
            selection['time_bin_start'] = [time_bin_start]
            selection['time_bin_end'] = [time_bin_end]
            selection = selection.stack(time_bin=('time_bin_start', 'time_bin_end'))
            return selection

        early = sel_time_bin(90, 110)
        late = sel_time_bin(190, 210)
        assembly = merge_data_arrays([early, late])
        return assembly


class ToliasCadena2017Loader(AssemblyLoader):
    def __init__(self):
        super(ToliasCadena2017Loader, self).__init__(name='tolias.Cadena2017')

    def __call__(self, average_repetition=True):
        assembly = brainscore.get_assembly(name='tolias.Cadena2017')
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly['region'] = 'neuroid_id', ['V1'] * len(assembly['neuroid_id'])
        assembly = assembly.stack(neuroid=['neuroid_id'])
        assembly = assembly.squeeze("time_bin")
        assembly = assembly.transpose('presentation', 'neuroid')
        if average_repetition:
            assembly = self.average_repetition(assembly)
        return assembly

    def average_repetition(self, assembly):
        attrs = assembly.attrs  # workaround to keeping attrs
        presentation_coords = [coord for coord, dims, values in walk_coords(assembly)
                               if array_is_element(dims, 'presentation')]
        presentation_coords = set(presentation_coords) - {'repetition_id', 'id'}
        assembly = assembly.multi_groupby(presentation_coords).mean(dim='presentation', skipna=True)
        assembly = assembly.dropna('presentation')  # discard any images with NaNs (~14%)
        assembly.attrs = attrs
        return assembly


class GallantDavid2004Loader(AssemblyLoader):
    def __init__(self):
        super(GallantDavid2004Loader, self).__init__(name='gallant.David2004')

    def __call__(self):
        assembly = brainscore.get_assembly(name=self.name)
        assembly.load()
        assembly = assembly.rename({'neuroid': 'neuroid_id'})
        assembly = assembly.stack(neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid')
        return assembly


metrics = {
    'rdm': lambda *args, **kwargs: NonparametricWrapper(RDMMetric(*args, **kwargs)),
    'linear_predictivity': LinearPredictivity,
    'pls_predictivity': PlsPredictivity,
    'edge_ratio': EdgeRatioMetric
}

assembly_loaders = [DicarloMajaj2015Loader(), DicarloMajaj2015EarlyLateLoader(), GallantDavid2004Loader(),
                    ToliasCadena2017Loader()]
assembly_loaders = {loader.name: loader for loader in assembly_loaders}

_benchmarks = {
    'brain-score': BrainScore,
    'dicarlo.Majaj2015': DicarloMajaj2015,
    'dicarlo.Majaj2015.earlylate': DicarloMajaj2015EarlyLate,
    'gallant.David2004': GallantDavid2004,
    'Felleman1991': FellemanVanEssen1991,
    'tolias.Cadena2017': ToliasCadena2017,
}


@cache()
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
