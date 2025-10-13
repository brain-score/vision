import brainscore_vision as bsv
def test_loads():
    assert bsv.load_model('vitpose_s_backbone').identifier == 'vitpose_s_backbone'
