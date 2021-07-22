import numpy as np
from brainio.assemblies import NeuroidAssembly
from pytest import approx

from brainscore.metrics.cka import CKACrossValidated, CKAMetric


class TestCKACrossValidated:
    def test_small(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        metric = CKACrossValidated()
        score = metric(assembly1=assembly, assembly2=assembly)
        assert score.sel(aggregation='center') == approx(1)


class TestCKAMetric:
    def test_equal30(self):
        assembly = NeuroidAssembly((np.arange(30 * 25) + np.random.standard_normal(30 * 25)).reshape((30, 25)),
                                   coords={'image_id': ('presentation', np.arange(30)),
                                           'object_name': ('presentation', ['a', 'b', 'c'] * 10),
                                           'neuroid_id': ('neuroid', np.arange(25)),
                                           'region': ('neuroid', [None] * 25)},
                                   dims=['presentation', 'neuroid'])
        similarity = CKAMetric()
        score = similarity(assembly, assembly)
        assert score == approx(1.)
