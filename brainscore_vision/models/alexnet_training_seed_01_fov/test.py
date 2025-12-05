import pytest
import brainscore_vision


@pytest.mark.travis_slow
@pytest.mark.parametrize('field_of_view', ['fov4', 'fov12', 'fov16'])
def test_has_identifier(field_of_view):
    model = brainscore_vision.load_model(f'alexnet_training_seed_01_{field_of_view}')
    assert model.identifier == f'alexnet_training_seed_01_{field_of_view}'
    