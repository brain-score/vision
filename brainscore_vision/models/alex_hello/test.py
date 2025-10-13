import brainscore_vision as bsv
def test_loads(): assert bsv.load_model('alex_hello').identifier == 'alex_hello'
