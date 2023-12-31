from string import ascii_lowercase as alphabet

import numpy as np
from pytest import approx

from brainio.assemblies import NeuroidAssembly, DataAssembly
from brainscore_vision import load_ceiling
from .ceiling import PearsonCorrelation


class TestInternalConsistency:
    def test_dummy_data(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'stimulus_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.repeat(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = load_ceiling('internal_consistency')
        ceiling = ceiler(data)
        assert ceiling == 1


class TestSplitHalfConsistency:
    def test(self):
        data = NeuroidAssembly(np.tile(np.arange(10)[:, np.newaxis], [5, 10]),
                               coords={'stimulus_id': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'image_meta': ('presentation', np.tile(list(alphabet)[:10], 5)),
                                       'repetition': ('presentation', np.tile(np.arange(5), 10)),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10))},
                               dims=['presentation', 'neuroid'])
        ceiler = PearsonCorrelation()
        ceiling = ceiler(data, data)
        assert all(ceiling == DataAssembly([approx(1)] * 10,
                                           coords={'neuroid_id': ('neuroid', np.arange(10)),
                                                   'neuroid_meta': ('neuroid', np.arange(10))},
                                           dims=['neuroid']))
