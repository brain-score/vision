import brainscore_vision


def test_has_identifier():
    model = brainscore_vision.load_model('20260408_stl10_disentangled_alpha00')
    assert model.identifier == '20260408_stl10_disentangled_alpha00'
