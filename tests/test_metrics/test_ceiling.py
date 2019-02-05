from pytest import approx
from string import ascii_lowercase as alphabet

import numpy as np
from brainio_base.assemblies import NeuroidAssembly, DataAssembly

from brainscore.benchmarks import DicarloMajaj2015Loader
from brainscore.metrics.ceiling import NoCeiling, InternalConsistency, SplitHalfConsistency


class TestNoCeiling:
    def test(self):
        ceiling = NoCeiling()
        ceiling_score = ceiling()
        assert ceiling_score == 1


class TestInternalConsistency:
    def test_dummy_data(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'image_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.repeat(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = InternalConsistency()
        ceiling = ceiler(data)
        assert ceiling.sel(aggregation='center') == 1

    def test_majaj2015_it(self):
        loader = DicarloMajaj2015Loader()
        assembly_repetitions = loader(average_repetition=False).sel(region='IT')
        ceiler = InternalConsistency()
        ceiling = ceiler(assembly_repetitions)
        assert ceiling.sel(aggregation='center') == approx(.82, abs=.01)


class TestSplitHalfConsistency:
    def test(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'image_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.tile(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = SplitHalfConsistency()
        ceiling = ceiler(data, data)
        assert all(ceiling == DataAssembly(np.ones(10),
                                           coords={'neuroid_id': ('neuroid', np.arange(10)),
                                                   'neuroid_meta': ('neuroid', np.arange(10))},
                                           dims=['neuroid']))
