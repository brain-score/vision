import pytest

from brainscore import get_stimulus_set
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers


@pytest.mark.memory_intense
class TestActivations:
    @pytest.mark.parametrize('model_identifier', [
        'alexnet',
        'resnet-101_v2',
        'mobilenet_v2_1.0_224',
        'resnet18-supervised',
        'resnet18-local_aggregation',
        'resnet18-instance_recognition',
        'resnet18-autoencoder',
        'resnet18-contrastive_predictive',
        'resnet18-colorization',
        'resnet18-relative_position',
        'resnet18-depth_prediction',
        'prednet',
        'resnet18-simclr',
        'resnet18-deepcluster',
        'resnet18-contrastive_multiview',
    ])
    def test_model(self, model_identifier):  # reset graph to get variable names back
        import keras
        keras.backend.clear_session()
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        layers = model_layers[model_identifier]
        activations_model = base_model_pool[model_identifier]
        stimulus_set = get_stimulus_set('dicarlo.hvm')
        stimulus_set = stimulus_set[:100]
        stimulus_set.identifier = 'dicarlo.hvm-min'
        activations = activations_model(stimulus_set, layers=layers)
        assert activations is not None
        assert hasattr(activations, 'layer')
