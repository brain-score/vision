import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('pr1_stl10_disentangled_alpha04')
    assert model.identifier == 'pr1_stl10_disentangled_alpha04'
