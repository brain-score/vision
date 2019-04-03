import numpy as np
import scipy.misc
from pytest import approx

import brainio_collection
from model_tools.brain_transformation import PixelsToDegrees


class TestPixelsToDegrees:
    def test_shape(self):
        stimulus_set = brainio_collection.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set['degrees'] = 8

        model_pixels = 224
        pixels_to_degrees = PixelsToDegrees(target_pixels=model_pixels)
        converted_stimuli = pixels_to_degrees(stimulus_set)

        non_degree_columns = list(set(stimulus_set.columns) - {'degrees'})
        assert len(converted_stimuli) == len(stimulus_set)
        assert converted_stimuli[non_degree_columns].equals(stimulus_set[non_degree_columns])  # equal metadata
        assert (converted_stimuli['degrees'] == 10).all()
        for image_id in converted_stimuli['image_id']:
            image_path = converted_stimuli.get_image(image_id)
            image = scipy.misc.imread(image_path)
            np.testing.assert_array_equal(image.shape, [224, 224, 3])

    def test_gray_background(self):
        stimulus_set = brainio_collection.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set = stimulus_set.loc[[0]]  # check only first image
        stimulus_set['degrees'] = 8

        model_pixels = 224
        pixels_to_degrees = PixelsToDegrees(target_pixels=model_pixels)
        converted_stimuli = pixels_to_degrees(stimulus_set)
        image_path = converted_stimuli.get_image(converted_stimuli['image_id'].values[0])
        image = scipy.misc.imread(image_path)

        amount_gray = 0
        for index in np.ndindex(image.shape[:2]):
            color = image[index]
            if (color == [128, 128, 128]).all():
                amount_gray += 1
        assert amount_gray / image.size == approx(.172041, abs=.0001)
        assert amount_gray == 25897

    def test_image_centered(self):
        stimulus_set = brainio_collection.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set = stimulus_set.loc[[0]]  # check only first image
        stimulus_set['degrees'] = 8

        model_pixels = 224
        pixels_to_degrees = PixelsToDegrees(target_pixels=model_pixels)
        converted_stimuli = pixels_to_degrees(stimulus_set)
        image_path = converted_stimuli.get_image(converted_stimuli['image_id'].values[0])
        image = scipy.misc.imread(image_path)

        gray = [128, 128, 128]
        assert (image[48, 48] == gray).all()
        assert (image[224 - 48, 224 - 48] == gray).all()
        assert (image[48, 224 - 48] == gray).all()
        assert (image[224 - 48, 48] == gray).all()

    def test_repeated_path(self):
        stimulus_set = brainio_collection.get_stimulus_set(name="dicarlo.hvm")
        stimulus_set = stimulus_set.loc[[0]]  # check only first image
        stimulus_set['degrees'] = 8

        model_pixels = 224
        pixels_to_degrees1 = PixelsToDegrees(target_pixels=model_pixels)
        converted_stimuli1 = pixels_to_degrees1(stimulus_set)
        image_path1 = converted_stimuli1.get_image(converted_stimuli1['image_id'].values[0])

        pixels_to_degrees2 = PixelsToDegrees(target_pixels=model_pixels)
        converted_stimuli2 = pixels_to_degrees2(stimulus_set)
        image_path2 = converted_stimuli2.get_image(converted_stimuli2['image_id'].values[0])

        assert image_path1 == image_path2
