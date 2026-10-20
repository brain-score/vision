import brainscore_vision

IDENTIFIER = 'alexnet_altahan_01'


def test_has_identifier():
    model = brainscore_vision.load_model(IDENTIFIER)
    assert model.identifier == IDENTIFIER


def test_commitment():
    model = brainscore_vision.load_model(IDENTIFIER)
    assert model.visual_degrees() == 8
    assert model.activations_model.image_size == 304
