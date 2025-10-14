import brainscore_vision as bsv
def test_loads():
    assert bsv.load_model('rtmpose_s_backbone').identifier == 'rtmpose_s_backbone'
