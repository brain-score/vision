import numpy as np

from brainio_base.assemblies import NeuroidAssembly, DataAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, \
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
            return DataAssembly(0)

    def test_presentation_neuroid(self):
        assembly = NeuroidAssembly(np.random.rand(500, 10),
                                   coords={'image_id': ('presentation', list(range(500))),
                                           'image_meta': ('presentation', [0] * 500),
                                           'neuroid_id': ('neuroid', list(range(10))),
                                           'neuroid_meta': ('neuroid', [0] * 10)},
                                   dims=['presentation', 'neuroid'])
        cv = CrossValidationSingle(splits=10, stratification_coord=None)
        metric = self.MetricPlaceholder()
        score = cv(assembly, apply=metric)
        assert len(metric.train_assemblies) == len(metric.test_assemblies) == 10
        assert len(score.attrs['raw']['split']) == 10

    def test_repeated_dim(self):
        """
        necessary for cross-validation over RDMs
        """
        assembly = NeuroidAssembly(np.random.rand(50, 50),
                                   coords={'image_id': ('presentation', list(range(50))),
                                           'image_meta': ('presentation', [0] * 50)},
                                   dims=['presentation', 'presentation'])
        cv = CrossValidationSingle(splits=10, train_size=.9, stratification_coord=None)
        metric = self.MetricPlaceholder()
        score = cv(assembly, apply=metric)
        assert len(metric.train_assemblies) == len(metric.test_assemblies) == 10
        assert all(assembly.shape == (45, 45) for assembly in metric.train_assemblies)
        assert all(assembly.shape == (5, 5) for assembly in metric.test_assemblies)
        assert len(score.attrs['raw']['split']) == 10


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
            return DataAssembly(0)

    def test_misaligned(self):
        jumbled_source = NeuroidAssembly(np.random.rand(500, 10),
                                         coords={'image_id': ('presentation', list(reversed(range(500)))),
                                                 'image_meta': ('presentation', [0] * 500),
                                                 'neuroid_id': ('neuroid', list(reversed(range(10)))),
                                                 'neuroid_meta': ('neuroid', [0] * 10)},
                                         dims=['presentation', 'neuroid'])
        target = jumbled_source.sortby(['image_id', 'neuroid_id'])
        cv = CrossValidation(splits=10, stratification_coord=None)
        metric = self.MetricPlaceholder()
        score = cv(jumbled_source, target, apply=metric)
        assert len(metric.train_source_assemblies) == len(metric.test_source_assemblies) == \
               len(metric.train_target_assemblies) == len(metric.test_target_assemblies) == 10
        assert len(score.attrs['raw']) == 10


class TestCartesianProduct:
    def test_no_division3(self):
        self._test_no_division_apply_manually(3)

    def test_no_division100(self):
        self._test_no_division_apply_manually(100)

    def _test_no_division_apply_manually(self, num_values):
        assembly = np.random.rand(num_values)
        assembly = NeuroidAssembly(assembly, coords={'neuroid': list(range(len(assembly)))}, dims=['neuroid'])
        transformation = CartesianProduct()
        generator = transformation.pipe(assembly)
        for divided_assembly in generator:  # should run only once
            np.testing.assert_array_equal(assembly.values, divided_assembly[0])
            done = generator.send(DataAssembly([0], coords={'split': [0]}, dims=['split']))
            assert done
            break
        similarity = next(generator)
        np.testing.assert_array_equal(similarity.shape, [1])
        np.testing.assert_array_equal(similarity.dims, ['split'])
        assert similarity[0] == 0

    class MetricPlaceholder(Metric):
        def __init__(self):
            super(TestCartesianProduct.MetricPlaceholder, self).__init__()
            self.assemblies = []

        def __call__(self, assembly):
            self.assemblies.append(assembly)
            return DataAssembly([0])

    def test_one_division(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(len(assembly))), 'division_coord': list(range(assembly.shape[1]))},
            dims=['neuroid', 'division_coord'])
        transformation = CartesianProduct(dividers=['division_coord'])
        placeholder = self.MetricPlaceholder()
        transformation(assembly, apply=placeholder)
        assert len(assembly['division_coord']) == len(placeholder.assemblies)
        targets = [assembly.sel(division_coord=i) for i in assembly['division_coord'].values]
        for target in targets:
            match = any([actual == target] for actual in placeholder.assemblies)
            assert match, "expected divided assembly not found: {target}"

    def test_one_division_similarity_dim_last(self):
        assembly = np.random.rand(3, 100)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(assembly.shape[1])), 'division_coord': list(range(assembly.shape[0]))},
            dims=['division_coord', 'neuroid'])
        transformation = CartesianProduct(dividers=['division_coord'])
        placeholder = self.MetricPlaceholder()
        transformation(assembly, apply=placeholder)
        assert len(assembly['division_coord']) == len(placeholder.assemblies)
        targets = [assembly.sel(division_coord=i) for i in assembly['division_coord'].values]
        for target in targets:
            match = any([actual == target] for actual in placeholder.assemblies)
            assert match, "expected divided assembly not found: {target}"

    def test_no_expand_raw_level(self):
        assembly = np.random.rand(3, 100)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(assembly.shape[1])), 'division_coord': list(range(assembly.shape[0]))},
            dims=['division_coord', 'neuroid'])
        transformation = CartesianProduct(dividers=['division_coord'])

        class RawMetricPlaceholder(Metric):
            def __call__(self, assembly, *args, **kwargs):
                result = Score([assembly.values[0]], dims=['dim'])
                raw = result.copy()
                raw['dim_id'] = 'dim', [assembly.values[1]]
                raw['division_coord'] = 'dim', [assembly.values[2]]
                result.attrs['raw'] = raw
                return result

        metric = RawMetricPlaceholder()
        result = transformation(assembly, apply=metric)
        assert hasattr(result, 'raw')
        assert 'division_coord' not in result.raw  # no dimension
        assert hasattr(result.raw, 'division_coord')  # but a level
