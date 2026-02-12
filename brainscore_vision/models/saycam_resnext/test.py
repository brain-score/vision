import brainscore_vision

def test_has_identifier():
    model = brainscore_vision.load_model('saycam_resnext')
    assert model.identifier == 'saycam_resnext'
