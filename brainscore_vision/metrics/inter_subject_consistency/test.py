from string import ascii_lowercase as alphabet

import numpy as np
from brainio.assemblies import NeuroidAssembly
from pytest import approx

from brainscore_vision import load_ceiling, load_metric
from numpy.random import RandomState


class TestInterSubjectConsistency:
    def test_dummy_data(self):
        rnd = RandomState(0)
        subject_matrix = rnd.rand(7, 5)
        data = NeuroidAssembly(np.concatenate([subject_matrix, subject_matrix], axis=1),
                               coords={'stimulus_id': ('presentation', list(alphabet)[:7]),
                                       'image_meta': ('presentation', list(alphabet)[:7]),
                                       'neuroid_id': ('neuroid', np.arange(10)),
                                       'neuroid_meta': ('neuroid', np.arange(10)),
                                       'subject': ('neuroid', np.repeat(['A', 'B'], 5))},
                               dims=['presentation', 'neuroid'])
        metric = load_metric('rdm')
        ceiler = load_ceiling('inter_subject_consistency', metric=metric)
        ceiling = ceiler(data)
        assert ceiling.item() == approx(1, abs=1e-8)
