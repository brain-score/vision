import itertools

import numpy as np

from mkgu.assemblies import NeuroidAssembly, DataAssembly
from mkgu.metrics import OuterCrossValidationSimilarity, NonparametricCVSimilarity


class SimilarityAdjacencyPlaceholder(OuterCrossValidationSimilarity):
    def __init__(self):
        super(SimilarityAdjacencyPlaceholder, self).__init__()
        self.source_assemblies = []
        self.target_assemblies = []

    def cross_apply(self, source_assembly, target_assembly):
        self.source_assemblies.append(source_assembly)
        self.target_assemblies.append(target_assembly)
        return DataAssembly([0], coords={'split': [0]}, dims=['split'])

    def apply_split(self, train_source, train_target, test_source, test_target, *args, **kwargs):
        raise NotImplementedError("should not be reached")


class TestSimilarityAdjacentProduct:
    def test_no_adjacent3(self):
        self._test_no_adjacent(3)

    def test_no_adjacent100(self):
        self._test_no_adjacent(100)

    def _test_no_adjacent(self, num_values):
        assembly = np.random.rand(num_values)
        assembly = NeuroidAssembly(assembly, coords={'neuroid': list(range(len(assembly)))}, dims=['neuroid'])
        similarity = SimilarityAdjacencyPlaceholder()
        similarity(assembly, assembly)
        assert 1 == len(similarity.source_assemblies) == len(similarity.target_assemblies)
        np.testing.assert_array_equal(assembly.values, similarity.source_assemblies[0])
        np.testing.assert_array_equal(assembly.values, similarity.target_assemblies[0])

    def test_one_adjacent(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(len(assembly))), 'adjacent_coord': list(range(assembly.shape[1]))},
            dims=['neuroid', 'adjacent_coord'])
        similarity = SimilarityAdjacencyPlaceholder()
        similarity(assembly, assembly)
        assert np.power(assembly.shape[1], 2) == len(similarity.source_assemblies) == len(similarity.target_assemblies)
        pairs = list(zip(similarity.source_assemblies, similarity.target_assemblies))
        target_pairs = [(assembly.sel(adjacent_coord=i).rename({'adjacent_coord': 'adjacent_coord-left'}).values,
                         assembly.sel(adjacent_coord=j).rename({'adjacent_coord': 'adjacent_coord-right'}).values)
                        for i, j in itertools.product(*([list(range(assembly.shape[1]))] * 2))]
        for source_values, target_values in target_pairs:
            match = False
            for source_actual, target_actual in pairs:
                if all(source_values == source_actual) and all(target_values == target_actual):
                    match = True
                    break
            assert match, "pair {} - {} not found".format(source_values, target_values)

    def test_one_adjacent_similarity_dim_last(self):
        assembly = np.random.rand(3, 100)
        assembly = NeuroidAssembly(
            assembly,
            coords={'neuroid': list(range(assembly.shape[1])), 'adjacent_coord': list(range(assembly.shape[0]))},
            dims=['adjacent_coord', 'neuroid'])
        similarity = SimilarityAdjacencyPlaceholder()
        similarity(assembly, assembly)
        assert np.power(assembly.shape[0], 2) == len(similarity.source_assemblies) == len(similarity.target_assemblies)
        pairs = list(zip(similarity.source_assemblies, similarity.target_assemblies))
        target_pairs = [(assembly.sel(adjacent_coord=i).rename({'adjacent_coord': 'adjacent_coord-left'}).values,
                         assembly.sel(adjacent_coord=j).rename({'adjacent_coord': 'adjacent_coord-right'}).values)
                        for i, j in itertools.product(*([list(range(assembly.shape[0]))] * 2))]
        for source_values, target_values in target_pairs:
            match = False
            for source_actual, target_actual in pairs:
                if all(source_values == source_actual) and all(target_values == target_actual):
                    match = True
                    break
            assert match, "pair {} - {} not found".format(source_values, target_values)


class SimilarityScorePlaceholder(NonparametricCVSimilarity):
    def __init__(self):
        super(SimilarityScorePlaceholder, self).__init__()
        self.source_assemblies = []
        self.target_assemblies = []

    def compute(self, source_assembly, target_assembly):
        self.source_assemblies.append(source_assembly)
        self.target_assemblies.append(target_assembly)
        return 0


class TestSimilarityScore:
    def test_presentation_x_neuroid(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(assembly, coords={
            'image_id': list(range(assembly.shape[0])),
            'neuroid_id': list(range(assembly.shape[1]))},
                                   dims=['image_id', 'neuroid_id'])
        assembly = assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))
        object_ratio = 10
        assembly['object_name'] = 'presentation', list(range(int(assembly.shape[0] / object_ratio))) * object_ratio
        similarity = SimilarityScorePlaceholder()
        score = similarity(assembly, assembly)
        assert 10 == len(similarity.source_assemblies) == len(similarity.target_assemblies)
        for assembly in similarity.source_assemblies + similarity.target_assemblies:
            assert len(assembly['presentation']) == 90
        assert all(score.values == 0)
        assert score.center == 0
