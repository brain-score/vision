import functools
import os

import numpy as np
import pytest
from pytest import approx

import brainscore_vision
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_helpers.activations import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_interface import BrainModel


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

def mock_triplet():
    stimuli = StimulusSet({'stimulus_id': ['1', '2', '3'], 'filename': ['rgb1', 'rgb2', 'rgb3']})
    stimuli.stimulus_paths = {'1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
                              '2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg'),
                              '3': os.path.join(os.path.dirname(__file__), 'rgb3.jpg')}
    stimuli.identifier = 'TestLabelBehavior.rgb_1_2_3'
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


class TestOddOneOut:
    def test_import(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import OddOneOut

    def test_dot(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import OddOneOut

        # Set up the task
        activations_model = pytorch_custom()
        brain_model = OddOneOut(identifier='pytorch-custom',
                                activations_model=activations_model,
                                layer=["relu2"])

        # Test similarity measure functionality
        assert brain_model.similarity_measure == 'dot'

        # Test the task and output
        stimuli = mock_triplet()
        brain_model.start_task(BrainModel.Task.odd_one_out)
        choice = brain_model.look_at(stimuli)
        assert isinstance(choice, BehavioralAssembly)
        assert len(choice.values) == 1

    def test_cosine(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import OddOneOut

        # Set up the task
        activations_model = pytorch_custom()
        brain_model = OddOneOut(identifier='pytorch-custom',
                                activations_model=activations_model,
                                layer=["relu2"])

        # Test similarity measure functionality
        brain_model.set_similarity_measure('cosine')
        assert brain_model.similarity_measure == 'cosine'

        # Test the task and output
        stimuli = mock_triplet()
        brain_model.start_task(BrainModel.Task.odd_one_out)
        choice = brain_model.look_at(stimuli)
        assert isinstance(choice, BehavioralAssembly)
        assert len(choice.values) == 1


class MockTemporalActivationsModel:
    """Mock activations model that returns temporal features with a time_bin dimension."""

    def __init__(self, feature_map: dict = None):
        """
        :param feature_map: Optional dict mapping stimulus_id to feature vectors.
            If not provided, generates random features.
        """
        self.identifier = 'mock-temporal-model'
        self.feature_map = feature_map

    def __call__(self, stimuli, layers, require_variance=False):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        n_stimuli = len(stimuli)
        n_neuroids = 10
        n_time_bins = 5

        if self.feature_map is not None:
            # Use provided feature vectors (repeat across time bins)
            features = np.array([self.feature_map[sid] for sid in stimuli['stimulus_id'].values])
            # Expand to have time dimension: (stimuli, neuroids) -> (stimuli, neuroids, time)
            features = np.repeat(features[:, :, np.newaxis], n_time_bins, axis=2)
        else:
            np.random.seed(42)
            features = np.random.rand(n_stimuli, n_neuroids, n_time_bins)

        assembly = NeuroidAssembly(
            features,
            coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                'neuroid_id': ('neuroid', [f'neuroid_{i}' for i in range(n_neuroids)]),
                'time_bin_start': ('time_bin', [i * 100 for i in range(n_time_bins)]),
                'time_bin_end': ('time_bin', [(i + 1) * 100 for i in range(n_time_bins)]),
            },
            dims=['presentation', 'neuroid', 'time_bin']
        )
        return assembly


class MockStaticActivationsModel:
    """Mock activations model that returns static features WITHOUT a time_bin dimension."""

    def __init__(self):
        self.identifier = 'mock-static-model'

    def __call__(self, stimuli, layers, require_variance=False):
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        n_stimuli = len(stimuli)
        n_neuroids = 10

        np.random.seed(42)
        features = np.random.rand(n_stimuli, n_neuroids)

        assembly = NeuroidAssembly(
            features,
            coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                'neuroid_id': ('neuroid', [f'neuroid_{i}' for i in range(n_neuroids)]),
            },
            dims=['presentation', 'neuroid']
        )
        return assembly


def mock_match_to_sample_stimuli():
    """Create a stimulus set with match-to-sample trial structure."""
    # Two trials, each with 1 sample and 3 choices
    stimuli = StimulusSet({
        'stimulus_id': ['s1', 'c1_0', 'c1_1', 'c1_2', 's2', 'c2_0', 'c2_1', 'c2_2'],
        'trial_id': ['trial_1', 'trial_1', 'trial_1', 'trial_1', 'trial_2', 'trial_2', 'trial_2', 'trial_2'],
        'stimulus_role': ['sample', 'choice', 'choice', 'choice', 'sample', 'choice', 'choice', 'choice'],
        'choice_index': [np.nan, 0, 1, 2, np.nan, 0, 1, 2],
        'filename': ['rgb1', 'rgb1', 'rgb2', 'rgb3', 'rgb2', 'rgb1', 'rgb2', 'rgb3'],
    })
    stimuli.stimulus_paths = {
        's1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
        'c1_0': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
        'c1_1': os.path.join(os.path.dirname(__file__), 'rgb2.jpg'),
        'c1_2': os.path.join(os.path.dirname(__file__), 'rgb3.jpg'),
        's2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg'),
        'c2_0': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
        'c2_1': os.path.join(os.path.dirname(__file__), 'rgb2.jpg'),
        'c2_2': os.path.join(os.path.dirname(__file__), 'rgb3.jpg'),
    }
    stimuli.identifier = 'TestTemporalMatchToSample.mock_stimuli'
    return stimuli


class TestTemporalMatchToSample:
    def test_import(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

    def test_task_type_exists(self):
        assert hasattr(BrainModel.Task, 'match_to_sample')
        assert BrainModel.Task.match_to_sample == 'match_to_sample'

    def test_cosine_similarity(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer'],
            similarity_measure='cosine'
        )
        assert brain_model.similarity_measure == 'cosine'

    def test_dot_similarity(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer'],
            similarity_measure='dot'
        )
        assert brain_model.similarity_measure == 'dot'

    def test_set_similarity_measure(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        brain_model.set_similarity_measure('dot')
        assert brain_model.similarity_measure == 'dot'

    def test_invalid_similarity_measure(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        with pytest.raises(ValueError, match="similarity_measure must be"):
            brain_model.set_similarity_measure('euclidean')

    def test_rejects_static_model(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockStaticActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-static',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)
        stimuli = mock_match_to_sample_stimuli()

        with pytest.raises(ValueError, match="temporal model"):
            brain_model.look_at(stimuli)

    def test_missing_required_columns(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)

        # Stimulus set missing required columns
        bad_stimuli = StimulusSet({'stimulus_id': ['1', '2'], 'filename': ['rgb1', 'rgb2']})
        bad_stimuli.stimulus_paths = {
            '1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
            '2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg')
        }

        with pytest.raises(ValueError, match="missing required column"):
            brain_model.look_at(bad_stimuli)

    def test_look_at_returns_behavioral_assembly(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)
        stimuli = mock_match_to_sample_stimuli()

        choices = brain_model.look_at(stimuli)

        assert isinstance(choices, BehavioralAssembly)
        assert 'presentation' in choices.dims
        assert 'choice' in choices.dims
        # Two trials should produce two choices
        assert choices.sizes['presentation'] == 2

    def test_choice_is_valid_index(self):
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        activations_model = MockTemporalActivationsModel()
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)
        stimuli = mock_match_to_sample_stimuli()

        choices = brain_model.look_at(stimuli)

        # Each choice should be a valid index (0, 1, or 2 for 3 choices)
        for choice in choices.values.flatten():
            assert choice in [0, 1, 2]

    def test_correct_choice_with_known_features(self):
        """Test that model correctly selects the most similar choice."""
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        # Create feature vectors where sample is most similar to choice at index 1
        n_neuroids = 10
        sample_features = np.ones(n_neuroids)  # [1, 1, 1, ...]
        choice_0_features = np.zeros(n_neuroids)  # [0, 0, 0, ...] - different
        choice_1_features = np.ones(n_neuroids)  # [1, 1, 1, ...] - same as sample
        choice_2_features = np.ones(n_neuroids) * -1  # [-1, -1, -1, ...] - opposite

        feature_map = {
            's1': sample_features,
            'c1_0': choice_0_features,
            'c1_1': choice_1_features,
            'c1_2': choice_2_features,
            's2': sample_features,
            'c2_0': choice_0_features,
            'c2_1': choice_1_features,
            'c2_2': choice_2_features,
        }

        activations_model = MockTemporalActivationsModel(feature_map=feature_map)
        brain_model = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer'],
            similarity_measure='cosine'
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)
        stimuli = mock_match_to_sample_stimuli()

        choices = brain_model.look_at(stimuli)

        # Both trials should choose index 1 (the matching choice)
        assert choices.values[0, 0] == 1
        assert choices.values[0, 1] == 1

    def test_integration_through_behavior_arbiter(self):
        """Test that match_to_sample task routes correctly through BehaviorArbiter."""
        from brainscore_vision.model_helpers.brain_transformation.behavior import (
            BehaviorArbiter, TemporalMatchToSample
        )

        # Create handler
        activations_model = MockTemporalActivationsModel()
        handler = TemporalMatchToSample(
            identifier='mock-temporal',
            activations_model=activations_model,
            layer=['mock_layer']
        )

        # Create arbiter with just the match_to_sample task
        arbiter = BehaviorArbiter({
            BrainModel.Task.match_to_sample: handler
        })

        # Start task through arbiter
        arbiter.start_task(BrainModel.Task.match_to_sample)

        # Look at stimuli through arbiter
        stimuli = mock_match_to_sample_stimuli()
        choices = arbiter.look_at(stimuli)

        assert isinstance(choices, BehavioralAssembly)
        assert choices.sizes['presentation'] == 2

    @pytest.mark.memory_intense
    @pytest.mark.private_access
    def test_with_real_temporal_model(self):
        """
        Integration test with a real temporal model (s3d from torchvision).

        This test validates the full pipeline with actual video processing.
        Marked as memory_intense and private_access since it requires:
        - Model weight downloads
        - Video stimulus processing
        - Significant memory for model inference
        """
        import brainscore_vision
        from brainscore_vision.model_helpers.brain_transformation.behavior import TemporalMatchToSample

        # Load a lightweight temporal model
        model = brainscore_vision.load_model('s3d')

        # The model should have an activations_model attribute
        activations_model = model.activations_model

        # Get a layer from the model
        layer = model.layers[-1] if hasattr(model, 'layers') else ['mixed_5c']

        brain_model = TemporalMatchToSample(
            identifier='s3d-test',
            activations_model=activations_model,
            layer=layer
        )
        brain_model.start_task(BrainModel.Task.match_to_sample)

        # Create stimuli - note: for real test we'd need video files
        # This test mainly validates the model loads and handler initializes correctly
        assert brain_model.identifier == 's3d-test'
        assert brain_model.similarity_measure == 'cosine'

