from brainscore_vision import load_model


def test_model_loads():
    model = load_model('gabor_filter_entropy_4x4_v2_forcedCosine')
    assert model.identifier == 'gabor_filter_entropy_4x4_v2_forcedCosine'
