import logging
import sys

import numpy as np
from pytest import approx

from mkgu.assemblies import NeuroidAssembly
from mkgu.metrics import NonparametricMetric, ParametricMetric

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class MetricScorePlaceholder(ParametricMetric):
    def apply(self, train_source, train_target, test_source, test_target):
        return np.random.normal(scale=0.01)


class TestSimilarityScoring:
    def test_one_division(self):
        assembly = np.random.rand(100, 3, 2)
        assembly = NeuroidAssembly(assembly, coords={
            'image_id': list(range(assembly.shape[0])),
            'neuroid_id': list(range(assembly.shape[1])),
            'division': list(range(assembly.shape[2]))},
                                   dims=['image_id', 'neuroid_id', 'division'])
        assembly = assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))
        assembly = assembly.transpose('presentation', 'neuroid', 'division')
        object_ratio = 10
        assembly['object_name'] = 'presentation', list(range(int(assembly.shape[0] / object_ratio))) * object_ratio
        similarity = MetricScorePlaceholder()
        sim = similarity(assembly, assembly)
        np.testing.assert_array_equal(sim.center.shape, [2, 2])
        assert (sim.center.values == approx(0, abs=0.1)).all()


class NonparametricPlaceholder(NonparametricMetric):
    def __init__(self):
        super(NonparametricPlaceholder, self).__init__()
        self.source_assemblies = []
        self.target_assemblies = []

    def compute(self, source_assembly, target_assembly):
        self.source_assemblies.append(source_assembly)
        self.target_assemblies.append(target_assembly)
        return 0


class TestNonparametric:
    def test_presentation_x_neuroid(self):
        assembly = np.random.rand(100, 3)
        assembly = NeuroidAssembly(assembly, coords={
            'image_id': list(range(assembly.shape[0])),
            'neuroid_id': list(range(assembly.shape[1]))},
                                   dims=['image_id', 'neuroid_id'])
        assembly = assembly.stack(presentation=('image_id',), neuroid=('neuroid_id',))
        object_ratio = 10
        assembly['object_name'] = 'presentation', list(range(int(assembly.shape[0] / object_ratio))) * object_ratio
        similarity = NonparametricPlaceholder()
        score = similarity(assembly, assembly)
        assert 10 == len(similarity.source_assemblies) == len(similarity.target_assemblies)
        for assembly in similarity.source_assemblies + similarity.target_assemblies:
            assert len(assembly['presentation']) == 90
        assert all(score.values == 0)
        assert score.center == 0
