import itertools

import numpy as np

from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics import Similarity


class SimilarityDummy(Similarity):
    def __init__(self):
        self.source_assemblies = []
        self.target_assemblies = []

    def apply(self, source_assembly, target_assembly):
        self.source_assemblies.append(source_assembly)
        self.target_assemblies.append(target_assembly)


class TestSimilarity:
    def test_no_adjacent3(self):
        self._test_no_adjacent(3)

    def test_no_adjacent100(self):
        self._test_no_adjacent(100)

    def _test_no_adjacent(self, num_values):
        assembly = np.random.rand(num_values)
        assembly = NeuroidAssembly(assembly, coords={'neuroid': list(range(len(assembly)))}, dims=['neuroid'])
        similarity = SimilarityDummy()
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
        similarity = SimilarityDummy()
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
        similarity = SimilarityDummy()
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
