import functools

import numpy as np
import pytest
import scipy.misc
from pytest import approx

import brainio_collection
from brainscore.benchmarks.loaders import load_assembly
from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment, PixelsToDegrees


def pytorch_custom():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.relu2(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=MyModel(), preprocessing=preprocessing)


class TestLayerSelection:
    @pytest.mark.parametrize(['model_ctr', 'layers', 'expected_layer', 'assembly_identifier', 'region'],
                             [(pytorch_custom, ['linear', 'relu2'], 'relu2', 'dicarlo.Majaj2015.IT', 'IT')])
    def test(self, model_ctr, layers, expected_layer, assembly_identifier, region):
        np.random.seed(0)
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, base_model=activations_model,
                                      layers=layers)
        assembly = load_assembly(assembly_identifier)
        brain_model.commit_region(region, assembly)

        brain_model.start_recording(region)
        predictions = brain_model.look_at(assembly.stimulus_set)
        assert set(predictions['region'].values) == {region}
        assert set(predictions['layer'].values) == {expected_layer}


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
