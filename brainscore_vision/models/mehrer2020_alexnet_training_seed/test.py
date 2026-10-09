import pytest

import brainscore_vision


@pytest.mark.travis_slow
@pytest.mark.parametrize('identifier', [
    'mehrer2020_alexnet_training_seed_01',
    'mehrer2020_alexnet_training_seed_02',
    'mehrer2020_alexnet_training_seed_03',
    'mehrer2020_alexnet_training_seed_04',
    'mehrer2020_alexnet_training_seed_05',
    'mehrer2020_alexnet_training_seed_06',
    'mehrer2020_alexnet_training_seed_07',
    'mehrer2020_alexnet_training_seed_08',
    'mehrer2020_alexnet_training_seed_09',
    'mehrer2020_alexnet_training_seed_10',
    'mehrer2020_alexnet_training_seed_01_fov4',
    'mehrer2020_alexnet_training_seed_01_fov12',
    'mehrer2020_alexnet_training_seed_01_fov16',
])
def test_has_identifier(identifier):
    model = brainscore_vision.load_model(identifier)
    assert model.identifier == identifier
