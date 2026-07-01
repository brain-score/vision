"""No-download coverage for vision activation assembly construction."""

from collections import OrderedDict

import numpy as np

from brainscore_vision.model_helpers.activations.core import (
    ActivationsExtractorHelper,
)


def _make_helper():
    helper = ActivationsExtractorHelper(
        get_activations=lambda *_args, **_kwargs: OrderedDict(),
        preprocessing=lambda x: x,
        identifier='vision-model',
    )
    for key in ('pixels', 'degrees'):
        helper._microsaccade_helper.microsaccades[key] = {
            'a.png': [(0.0, 0.0)],
            'b.png': [(0.0, 0.0)],
        }
    return helper


def test_activation_helper_package_uses_shared_builder():
    helper = _make_helper()
    activations = OrderedDict([
        ('layer1', np.ones((2, 2), dtype=np.float32)),
        ('layer2', np.zeros((2, 1), dtype=np.float32)),
    ])

    assembly = helper._package(
        layer_activations=activations,
        stimuli_paths=['a.png', 'b.png'],
        require_variance=False,
    )

    assert assembly.dims == ('presentation', 'neuroid')
    assert assembly.shape == (2, 3)
    assert list(assembly['stimulus_path'].values) == ['a.png', 'b.png']
    assert list(assembly['neuroid_id'].values) == [
        'vision-model.layer1.0',
        'vision-model.layer1.1',
        'vision-model.layer2.0',
    ]
    assert list(assembly['layer'].values) == ['layer1', 'layer1', 'layer2']
