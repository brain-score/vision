import brainscore_vision

IDENTIFIER = 'vit_large_patch14_clip_altahan_02'


def test_has_identifier():
    model = brainscore_vision.load_model(IDENTIFIER)
    assert model.identifier == IDENTIFIER


def test_commitment():
    model = brainscore_vision.load_model(IDENTIFIER)
    assert model.visual_degrees() == 8
    assert model.activations_model.image_size == 224
