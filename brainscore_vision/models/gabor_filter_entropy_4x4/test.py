import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('gabor_filter_entropy_4x4')
    assert model.identifier == 'gabor_filter_entropy_4x4'


def test_has_default_layer():
    model = brainscore_vision.load_model('gabor_filter_entropy_4x4')
    assert model.layers == ['gabor_entropy_4x4']
