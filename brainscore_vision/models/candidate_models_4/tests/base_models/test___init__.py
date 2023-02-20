import os

import pytest

from candidate_models.base_models import base_model_pool


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_name', list(base_model_pool.keys()))
def test_run_logits(model_name):
    base_model = base_model_pool[model_name]
    try:
        # up until here, the model is just a LazyLoad object that is only initialized upon attribute access.
        activations = base_model([os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=None)
        assert activations is not None
    finally:
        # reset graph to get variable names back
        import keras
        keras.backend.clear_session()
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize(['model_name', 'expected_identifier'], [
    ('alexnet', 'alexnet'),
    ('resnet-34', 'resnet34'),
    ('densenet-169', 'densenet169'),
])
def test_identifier(model_name, expected_identifier):
    model = base_model_pool[model_name]
    assert model.identifier == expected_identifier
