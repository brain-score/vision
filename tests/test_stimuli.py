import numpy as np
import scipy.misc
from pytest import approx

import brainscore


def test_loadname_dicarlo_hvm():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm") is not None


def test_loadname_dicarlo_hvm_v0():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v0") is not None


def test_loadname_dicarlo_hvm_v3():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v3") is not None


def test_loadname_dicarlo_hvm_v6():
    assert brainscore.get_stimulus_set(name="dicarlo.hvm.v6") is not None


class TestLoadImage:
    def test_dicarlohvm(self):
        stimulus_set = brainscore.get_stimulus_set(name="dicarlo.hvm")
        paths = stimulus_set.image_paths.values()
        for path in paths:
            image = scipy.misc.imread(path)
            assert isinstance(image, np.ndarray)
            assert image.size > 0

    def test_shape(self):
        stimulus_set = brainscore.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set['degrees'] = [8] * len(stimulus_set)
        image_id = stimulus_set.image_id[0]
        image_path = stimulus_set.get_image(image_id, pixels_central_vision=(224, 224))
        image = scipy.misc.imread(image_path)
        np.testing.assert_array_equal(image.shape, [224, 224, 3])

    def test_gray_background(self):
        stimulus_set = brainscore.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set['degrees'] = [8] * len(stimulus_set)
        image_id = stimulus_set.image_id[0]
        image_path = stimulus_set.get_image(image_id, pixels_central_vision=(224, 224))
        image = scipy.misc.imread(image_path)
        amount_gray = 0
        for index in np.ndindex(image.shape[:2]):
            color = image[index]
            if (color == [128, 128, 128]).all():
                amount_gray += 1
        assert amount_gray / image.size == approx(.172041, abs=.0001)
        assert amount_gray == 25897
