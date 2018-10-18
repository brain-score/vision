import xarray as xr
import itertools

import numpy as np
import pytest

from brainscore import benchmarks
from brainscore.assemblies import NeuroidAssembly, DataAssembly
from brainscore.metrics import Metric
from brainscore.metrics.transformations import subset, index_efficient, CartesianProduct, CrossValidation, \
    CrossValidationSingle


class TestCrossValidationSingle:
    class MetricPlaceholder(Metric):
        def __init__(self):
            super(TestCrossValidationSingle.MetricPlaceholder, self).__init__()
            self.train_assemblies = []
            self.test_assemblies = []

        def __call__(self, train, test):
            self.train_assemblies.append(train)
            self.test_assemblies.append(test)
            return DataAssembly([0])

    def test(self):
        assembly = NeuroidAssembly(np.random.rand(500, 10),
                                   coords={'image_id': ('presentation', list(range(500))),
                                           'image_meta': ('presentation', [0] * 500),
                                           'neuroid_id': ('neuroid', list(range(10))),
                                           'neuroid_meta': ('neuroid', [0] * 10)},
                                   dims=['presentation', 'neuroid'])
        cv = CrossValidationSingle(splits=10)
        metric = self.MetricPlaceholder()
        score = cv(assembly, apply=metric)
        assert len(metric.train_assemblies) == len(metric.test_assemblies) == 10
        assert len(score.values['split']) == 10


class TestCrossValidation:
    class MetricPlaceholder(Metric):
        def __init__(self):
            super(TestCrossValidation.MetricPlaceholder, self).__init__()
            self.train_source_assemblies = []
            self.test_source_assemblies = []
            self.train_target_assemblies = []
            self.test_target_assemblies = []

        def __call__(self, train_source, train_target, test_source, test_target):
            assert sorted(train_source['image_id'].values) == sorted(train_target['image_id'].values)
            assert sorted(test_source['image_id'].values) == sorted(test_target['image_id'].values)
            self.train_source_assemblies.append(train_source)
            self.train_target_assemblies.append(train_target)
            self.test_source_assemblies.append(test_source)
            self.test_target_assemblies.append(test_target)
            return DataAssembly([0])

    def test_misaligned(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10)},
                                         dims=['presentation', 'neuroid'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        cv = CrossValidation(splits=10)
        metric = self.MetricPlaceholder()
        score = cv(jumbled_source, target, apply=metric)
        assert len(metric.train_source_assemblies) == len(metric.test_source_assemblies) == \
               len(metric.train_target_assemblies) == len(metric.test_target_assemblies) == 10
        assert len(score.values['split']) == 10


class TestCartesianProduct:
    class MetricPlaceholder(Metric):
        def __init__(self):
            super(TestCartesianProduct.MetricPlaceholder, self).__init__()
            self.source_assemblies = []
            self.target_assemblies = []

        def __call__(self, source_assembly, target_assembly):
            self.source_assemblies.append(source_assembly)
            self.target_assemblies.append(target_assembly)
            return DataAssembly([0])

    def test_no_division3(self):
        self._test_no_division_apply_manually(3)

    def test_no_division100(self):
        self._test_no_division_apply_manually(100)

    def _test_no_division_apply_manually(self, num_values):
        assembly = np.random.rand(num_values)
        assembly = NeuroidAssembly(assembly, coords={'neuroid': list(range(len(assembly)))}, dims=['neuroid'])
        transformation = CartesianProduct()
        generator = transformation.pipe(assembly, assembly)
        for source, target in generator:  # should run only once
            np.testing.assert_array_equal(assembly.values, source)
            np.testing.assert_array_equal(assembly.values, target)
            done = generator.send(DataAssembly([0], coords={'split': [0]}, dims=['split']))
            assert done
            break
        similarity = next(generator)
        np.testing.assert_array_equal(similarity.shape, [1])
        np.testing.assert_array_equal(similarity.dims, ['split'])
        assert similarity[0] == 0

    def test_one_division(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(len(assembly))), 'division_coord': list(range(assembly.shape[1]))},
            dims=['neuroid', 'division_coord'])
        transformation = CartesianProduct()
        placeholder = self.MetricPlaceholder()
        transformation(assembly, assembly, apply=placeholder)
        assert np.power(assembly.shape[1], 2) == \
               len(placeholder.source_assemblies) == len(placeholder.target_assemblies)
        pairs = list(zip(placeholder.source_assemblies, placeholder.target_assemblies))
        target_pairs = [(assembly.sel(division_coord=i).rename({'division_coord': 'division_coord-source'}).values,
                         assembly.sel(division_coord=j).rename({'division_coord': 'division_coord-target'}).values)
                        for i, j in itertools.product(*([list(range(assembly.shape[1]))] * 2))]
        for source_values, target_values in target_pairs:
            match = False
            for source_actual, target_actual in pairs:
                if all(source_values == source_actual) and all(target_values == target_actual):
                    match = True
                    break
            assert match, "pair {} - {} not found".format(source_values, target_values)

    def test_one_division_similarity_dim_last(self):
        assembly = np.random.rand(3, 100)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(assembly.shape[1])), 'division_coord': list(range(assembly.shape[0]))},
            dims=['division_coord', 'neuroid'])
        transformation = CartesianProduct()
        placeholder = self.MetricPlaceholder()
        transformation(assembly, assembly, apply=placeholder)
        assert np.power(assembly.shape[0], 2) == len(placeholder.source_assemblies) == len(
            placeholder.target_assemblies)
        pairs = list(zip(placeholder.source_assemblies, placeholder.target_assemblies))
        target_pairs = [(assembly.sel(division_coord=i).rename({'division_coord': 'division_coord-source'}).values,
                         assembly.sel(division_coord=j).rename({'division_coord': 'division_coord-target'}).values)
                        for i, j in itertools.product(*([list(range(assembly.shape[0]))] * 2))]
        for source_values, target_values in target_pairs:
            match = False
            for source_actual, target_actual in pairs:
                if all(source_values == source_actual) and all(target_values == target_actual):
                    match = True
                    break
            assert match, "pair {} - {} not found".format(source_values, target_values)


class TestSubset:
    def test_equal(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(assembly, coords={
            'image_id': list(range(assembly.shape[0])),
            'neuroid_id': list(range(assembly.shape[1]))},
                                   dims=['image_id', 'neuroid_id'])
        assembly = assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))
        subset_assembly = subset(assembly, assembly, subset_dims=('presentation',))
        assert (subset_assembly == assembly).all()

    def test_equal_shifted(self):
        target_assembly = np.random.rand(100, 3)
        target_assembly = NeuroidAssembly(target_assembly, coords={
            'image_id': list(range(target_assembly.shape[0])),
            'neuroid_id': list(range(target_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        target_assembly = target_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        shifted_values = np.concatenate((target_assembly.values[1:], target_assembly.values[:1]))
        shifed_ids = np.array(list(range(shifted_values.shape[0]))) + 1
        shifed_ids[-1] = 0
        source_assembly = NeuroidAssembly(shifted_values, coords={
            'image_id': shifed_ids,
            'neuroid_id': list(range(shifted_values.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        source_assembly = source_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        subset_assembly = subset(source_assembly, target_assembly, subset_dims=('presentation',))
        np.testing.assert_array_equal(subset_assembly.coords.keys(), target_assembly.coords.keys())
        assert subset_assembly.shape == target_assembly.shape

    def test_smaller_first(self):
        source_assembly = np.random.rand(100, 3)
        source_assembly = NeuroidAssembly(source_assembly, coords={
            'image_id': list(range(source_assembly.shape[0])),
            'neuroid_id': list(range(source_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        source_assembly = source_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        target_assembly = source_assembly.sel(presentation=list(map(lambda x: (x,), range(50))),
                                              neuroid=list(map(lambda x: (x,), range(2))))

        subset_assembly = subset(source_assembly, target_assembly, subset_dims=('presentation',))
        np.testing.assert_array_equal(subset_assembly.coords.keys(), target_assembly.coords.keys())
        for coord_name in target_assembly.coords:
            assert all(subset_assembly[coord_name] == target_assembly[coord_name])
        assert (subset_assembly == target_assembly).all()

    def test_smaller_last(self):
        source_assembly = np.random.rand(100, 3)
        source_assembly = NeuroidAssembly(source_assembly, coords={
            'image_id': list(range(source_assembly.shape[0])),
            'neuroid_id': list(range(source_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        source_assembly = source_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        target_assembly = source_assembly.sel(presentation=list(map(lambda x: (50 + x,), range(50))),
                                              neuroid=list(map(lambda x: (1 + x,), range(2))))

        subset_assembly = subset(source_assembly, target_assembly, subset_dims=('presentation',))
        np.testing.assert_array_equal(subset_assembly.coords.keys(), target_assembly.coords.keys())
        for coord_name in target_assembly.coords:
            assert all(subset_assembly[coord_name] == target_assembly[coord_name])
        assert (subset_assembly == target_assembly).all()

    def test_larger_error(self):
        source_assembly = np.random.rand(50, 2)
        source_assembly = NeuroidAssembly(source_assembly, coords={
            'image_id': list(range(source_assembly.shape[0])),
            'neuroid_id': list(range(source_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        source_assembly = source_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        target_assembly = np.random.rand(100, 3)
        target_assembly = NeuroidAssembly(target_assembly, coords={
            'image_id': list(range(target_assembly.shape[0])),
            'neuroid_id': list(range(target_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        target_assembly = target_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        with pytest.raises(Exception):
            subset(source_assembly, target_assembly, subset_dims=('presentation',))

    def test_repeated_target(self):
        source_assembly = np.random.rand(5, 3)
        source_assembly = NeuroidAssembly(source_assembly, coords={
            'image_id': list(range(source_assembly.shape[0])),
            'neuroid_id': list(range(source_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        source_assembly = source_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        target_assembly = NeuroidAssembly(np.repeat(source_assembly, 2, axis=0), coords={
            'image_id': np.repeat(list(range(source_assembly.shape[0])), 2, axis=0),
            'neuroid_id': list(range(source_assembly.shape[1]))},
                                          dims=['image_id', 'neuroid_id'])
        target_assembly = target_assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))

        subset_assembly = subset(source_assembly, target_assembly, subset_dims=('presentation',), repeat=True)
        np.testing.assert_array_equal(subset_assembly.coords.keys(), target_assembly.coords.keys())
        for coord_name in target_assembly.coords:
            assert all(subset_assembly[coord_name] == target_assembly[coord_name])
        np.testing.assert_array_equal(subset_assembly, target_assembly)
        assert (subset_assembly == target_assembly).all()

    def test_category_subselection(self):
        assembly = benchmarks.load_assembly('dicarlo.Majaj2015')
        categories = np.unique(assembly['category_name'])
        target = xr.DataArray([0] * len(categories), coords={'category_name': categories},
                              dims=['category_name']).stack(presentation=['category_name'])
        sub_assembly = subset(assembly, target, repeat=True, dims_must_match=False)
        assert (assembly == sub_assembly).all()


class TestIndexEfficient:
    def test(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 1, 3, 4, 4, 4, 5])
        indexer = [a.tolist().index(target_val) for target_val in b]
        indexer = [index for index in indexer if index != -1]
        result = index_efficient(a, b)
        assert result == indexer
