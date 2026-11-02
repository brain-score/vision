"""`number_of_trials` as part of the stored-activations cache key.

With `require_variance=True`, the model runs every stimulus `number_of_trials`
times (once per microsaccade), so results for different trial counts must not
share a cache entry. Without `require_variance` every stimulus runs once, and
the key must stay what it was, so existing caches remain valid.
"""
import functools
import os
import unittest.mock as mock

import pytest
import torch

from brainscore_vision.model_helpers.activations import PytorchWrapper
from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images


def number_of_trials_passed_to_the_stored_call(require_variance, number_of_trials):
    extractor = ActivationsExtractorHelper(get_activations=mock.Mock(), preprocessing=None, identifier='alexnet')
    extractor._microsaccade_helper.number_of_trials = number_of_trials  # what `__call__` sets
    extractor._from_paths_stored = mock.Mock(return_value='ASSEMBLY')
    result = extractor.from_paths(stimuli_paths=['x.png'], layers=['fc'], stimuli_identifier='hvm-public',
                                  require_variance=require_variance)
    assert result == 'ASSEMBLY', "the returned assembly must be untouched"
    # not passing it is the same as passing the default, 1
    return extractor._from_paths_stored.call_args[1].get('number_of_trials', 1)


@pytest.mark.parametrize('number_of_trials', [1, 2, 5])
def test_number_of_trials_reaches_the_stored_call(number_of_trials):
    assert number_of_trials_passed_to_the_stored_call(
        require_variance=True, number_of_trials=number_of_trials) == number_of_trials


@pytest.mark.parametrize('number_of_trials', [1, 3])
def test_deterministic_key_is_unchanged(number_of_trials):
    assert number_of_trials_passed_to_the_stored_call(
        require_variance=False, number_of_trials=number_of_trials) == 1


def test_cached_activations_match_the_requested_number_of_trials(tmp_path, monkeypatch):
    """Two scoring runs of the same model on the same stimuli, with different trial counts."""
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    for variable in ['RESULTCACHING_CACHEDONLY', 'RESULTCACHING_FORMAT', 'RESULTCACHING_S3_BUCKET',
                     'BRAINSCORE_CACHE_PLUGIN_REVISION']:
        monkeypatch.delenv(variable, raising=False)
    # `RESULTCACHING_HOME` is read when the decorator is created, so point the existing storage at `tmp_path`
    storage = next(cell.cell_contents for cell in ActivationsExtractorHelper._from_paths_stored.__closure__
                   if hasattr(cell.cell_contents, 'identifier_ignore'))
    monkeypatch.setattr(storage, '_storage_directory', str(tmp_path))
    stimulus_paths = [os.path.join(os.path.dirname(__file__), 'grayscale2.jpg')]

    for number_of_trials in [2, 5]:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
        activations_extractor = PytorchWrapper(model=model, identifier='number-of-trials-cache-test',
                                               preprocessing=functools.partial(load_preprocess_images,
                                                                               image_size=32),
                                               batch_size=1)
        activations_extractor._extractor.set_visual_degrees(8.)
        activations = activations_extractor(stimuli=stimulus_paths, layers=['0'],
                                            stimuli_identifier='number-of-trials-cache-test-stimuli',
                                            number_of_trials=number_of_trials, require_variance=True)
        assert len(activations['presentation']) == number_of_trials

    stored = [name for _, _, files in os.walk(tmp_path) for name in files]
    assert {n for n in [2, 5] if any(f'number_of_trials={n},' in name for name in stored)} == {2, 5}, stored
