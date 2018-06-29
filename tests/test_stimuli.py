import numpy as np
import scipy.misc

import brainscore


def test_loadname_dicarlo_hvm():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm") is not None


def test_loadname_dicarlo_hvm_v0():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v0") is not None


def test_loadname_dicarlo_hvm_v3():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v3") is not None


def test_loadname_dicarlo_hvm_v6():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v6") is not None


def test_load_images():
    stimulus_set = brainscore.get_stimulus_set(name="dicarlo.hvm")
    paths = stimulus_set.image_paths.values()
    for path in paths:
        image = scipy.misc.imread(path)
        assert isinstance(image, np.ndarray)
        assert image.size > 0
