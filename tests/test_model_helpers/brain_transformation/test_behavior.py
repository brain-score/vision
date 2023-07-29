import functools
import numpy as np
import os
import pytest
import xarray as xr
from pathlib import Path
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet
from brainscore_vision.benchmarks.rajalingham2018 import _DicarloRajalingham2018
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.metrics.image_level_behavior import I2n
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.model_helpers.activations import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment, ProbabilitiesMapping


def pytorch_custom():
    import torch
    from torch import nn
    from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            np.random.seed(0)
            torch.random.manual_seed(0)
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


class TestLabelBehavior:
    @pytest.mark.parametrize(['model_ctr'], [(pytorch_custom,)])
    def test_imagenet_creates_synset(self, model_ctr):
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='dummy')  # not needed
        stimuli = mock_stimulus_set()
        brain_model.start_task(BrainModel.Task.label, 'imagenet')
        behavior = brain_model.look_at(stimuli)
        assert isinstance(behavior, BehavioralAssembly)
        assert set(behavior['stimulus_id'].values) == {'1', '2'}
        assert len(behavior['synset']) == 2
        assert behavior['synset'].values[0].startswith('n')

    @pytest.mark.parametrize(['model_ctr'], [(pytorch_custom,)])
    def test_choicelabels(self, model_ctr):
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=['relu1', 'relu2'], behavioral_readout_layer='relu2')
        stimuli = mock_stimulus_set()
        choice_labels = ['dog', 'cat', 'bear', 'bird']
        brain_model.start_task(BrainModel.Task.label, choice_labels)
        behavior = brain_model.look_at(stimuli)
        assert isinstance(behavior, BehavioralAssembly)
        assert set(behavior['stimulus_id'].values) == {'1', '2'}
        assert all(choice in choice_labels for choice in behavior.squeeze().values)
        # these two labels do not necessarily make sense since we're working with a random model
        assert behavior.sel(stimulus_id='1').values.item() == 'bear'
        assert behavior.sel(stimulus_id='2').values.item() == 'bird'


def mock_stimulus_set():
    stimuli = StimulusSet({'stimulus_id': ['1', '2'], 'filename': ['rgb1', 'rgb2']})
    stimuli.stimulus_paths = {'1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
                              '2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg')}
    stimuli.identifier = 'TestLabelBehavior.rgb_1_2'
    return stimuli


class TestLogitsBehavior:
    """
    legacy support for LogitsBehavior; still used in old candidate_models submissions
    https://github.com/brain-score/candidate_models/blob/fa965c452bd17c6bfcca5b991fdbb55fd5db618f/candidate_models/model_commitments/cornets.py#L13
    """

    def test_import(self):
        # noinspection PyUnresolvedReferences
        from brainscore_vision.model_helpers.brain_transformation.behavior import LogitsBehavior

    def test_imagenet(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import LogitsBehavior

        activations_model = pytorch_custom()
        logits_behavior = LogitsBehavior(
            identifier='pytorch-custom', activations_model=activations_model)

        stimuli = mock_stimulus_set()
        logits_behavior.start_task(BrainModel.Task.label, 'imagenet')
        behavior = logits_behavior.look_at(stimuli)
        assert isinstance(behavior, BehavioralAssembly)
        assert set(behavior['stimulus_id'].values) == {'1', '2'}
        assert len(behavior['synset']) == 2
        assert behavior['synset'].values[0].startswith('n')


class TestProbabilitiesMapping:
    def test_creates_probabilities(self):
        activations_model = pytorch_custom()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='relu2')
        fitting_stimuli = StimulusSet({'stimulus_id': ['rgb1', 'rgb2'], 'image_label': ['label1', 'label2']})
        fitting_stimuli.stimulus_paths = {'rgb1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
                                          'rgb2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg')}
        fitting_stimuli.identifier = 'test_probabilities_mapping.creates_probabilities'
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=brain_model.visual_degrees(),
                                          source_visual_degrees=8)
        brain_model.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        probabilities = brain_model.look_at(fitting_stimuli)
        np.testing.assert_array_equal(probabilities.dims, ['presentation', 'choice'])
        np.testing.assert_array_equal(probabilities.shape, [2, 2])
        np.testing.assert_array_almost_equal(probabilities.sel(stimulus_id='rgb1', choice='label1').values,
                                             probabilities.sel(stimulus_id='rgb2', choice='label2').values)
        assert probabilities.sel(stimulus_id='rgb1', choice='label1') + \
               probabilities.sel(stimulus_id='rgb1', choice='label2') == approx(1)


@pytest.mark.private_access
class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        class UnceiledBenchmark(_DicarloRajalingham2018):
            def __init__(self):
                super(UnceiledBenchmark, self).__init__(metric=I2n(), metric_identifier='i2n')

            def __call__(self, candidate: BrainModel):
                candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)
                probabilities = candidate.look_at(self._assembly.stimulus_set)
                score = self._metric(probabilities, self._assembly)
                return score

        benchmark = UnceiledBenchmark()
        # features
        path_to_expected = Path(__file__).parent / f'identifier={model},stimuli_identifier=objectome-240.nc'
        feature_responses = xr.load_dataarray(path_to_expected)
        feature_responses['stimulus_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                             for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer

        class PrecomputedFeatures:
            def __init__(self, precomputed_features):
                self.features = precomputed_features

            def __call__(self, stimuli, layers):
                np.testing.assert_array_equal(layers, ['behavioral-layer'])
                self_stimulus_ids = self.features['stimulus_id'].values.tolist()
                indices = [self_stimulus_ids.index(stimulus_id) for stimulus_id in stimuli['stimulus_id'].values]
                features = self.features[{'presentation': indices}]
                return features

        # evaluate candidate
        transformation = ProbabilitiesMapping(identifier=f'TestI2N.{model}',
                                              activations_model=PrecomputedFeatures(feature_responses),
                                              layer='behavioral-layer')
        score = benchmark(transformation)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"
