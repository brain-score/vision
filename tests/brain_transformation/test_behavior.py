import functools
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

from brainio_base.stimuli import StimulusSet
from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset
from brainscore.model_interface import BrainModel
from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment, ProbabilitiesMapping


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


class TestLogitsBehavior:
    @pytest.mark.parametrize(['model_ctr'], [(pytorch_custom,)])
    def test_creates_synset(self, model_ctr):
        np.random.seed(0)
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='dummy')  # not needed
        stimuli = StimulusSet({'image_id': ['abc123']})
        stimuli.image_paths = {'abc123': os.path.join(os.path.dirname(__file__), 'rgb1.jpg')}
        stimuli.name = 'test_logits_behavior.creates_synset'
        brain_model.start_task(BrainModel.Task.label)
        synsets = brain_model.look_at(stimuli)
        assert len(synsets) == 1
        assert synsets[0].startswith('n')


class TestProbabilitiesMapping:
    def test_creates_probabilities(self):
        activations_model = pytorch_custom()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='relu2')
        fitting_stimuli = StimulusSet({'image_id': ['rgb1', 'rgb2'], 'label': ['label1', 'label2']})
        fitting_stimuli.image_paths = {'rgb1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
                                       'rgb2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg')}
        fitting_stimuli.name = 'test_probabilities_mapping.creates_probabilities'
        brain_model.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        probabilities = brain_model.look_at(fitting_stimuli)
        np.testing.assert_array_equal(probabilities.dims, ['presentation', 'choice'])
        np.testing.assert_array_equal(probabilities.shape, [2, 2])
        assert probabilities.sel(image_id='rgb1', choice='label1').values == \
               probabilities.sel(image_id='rgb2', choice='label2').values
        assert probabilities.sel(image_id='rgb1', choice='label1') + \
               probabilities.sel(image_id='rgb1', choice='label2') == 1


class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        # assembly
        fitting_objectome, testing_objectome = self.get_objectome('partial_trials'), self.get_objectome('full_trials')

        # features
        feature_responses = pd.read_pickle(
            os.path.join(os.path.dirname(__file__), f'identifier={model},stimuli_identifier=objectome-240.pkl'))['data']
        feature_responses['image_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                          for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        expected_images = set(fitting_objectome['image_id'].values) | set(testing_objectome['image_id'].values)
        assert expected_images.issuperset(set(feature_responses['image_id'].values))
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer

        class PrecomputedFeatures:
            def __init__(self, precomputed_features):
                self.features = precomputed_features

            def __call__(self, stimuli, layers):
                np.testing.assert_array_equal(layers, ['behavioral-layer'])
                image_ids = stimuli['image_id'].values
                image_ids = xr.DataArray(np.zeros(len(image_ids)), coords={'image_id': image_ids},
                                         dims=['image_id']).stack(presentation=['image_id'])
                features = subset(self.features, image_ids)
                return features

        # transform
        transformation = ProbabilitiesMapping(identifier=f'TestI2N.{model}',
                                              activations_model=PrecomputedFeatures(feature_responses),
                                              layer='behavioral-layer')
        transformation.start_task(BrainModel.Task.probabilities, fitting_objectome.stimulus_set)
        testing_features = transformation.look_at(testing_objectome.stimulus_set)
        with open(f'/braintree/home/msch/brain-score/tests/test_metrics/{model}-transformed_features.pkl', 'wb') as f:
            feats = xr.DataArray(testing_features).reset_index(['presentation', 'choice'])
            pickle.dump({'data': feats}, f)
        # metric
        i2n = I2n()
        score = i2n(testing_features, testing_objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def get_objectome(self, subtype):
        # TODO: remove once packaged
        basepath = '/braintree/home/msch/brainio_contrib/mkgu_packaging/dicarlo'
        with open(f'{basepath}/dicarlo.Rajalingham2018.{subtype}.pkl',
                  'rb') as f:
            objectome = pickle.load(f)
        with open(f'{basepath}/dicarlo.Rajalingham2018.{subtype}-stim.pkl',
                  'rb') as f:
            stimulus_set = pickle.load(f)
        objectome.attrs['stimulus_set'] = stimulus_set
        return objectome
