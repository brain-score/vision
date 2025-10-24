import pytest
import brainscore_vision

def test_has_identifier():
    model = brainscore_vision.load_model('chresmax_v3_lambda.ip_3_chresmax_v3_gpu_8_cl_0.01_ip_3_322_322_18432_c1[_6*3*1_]_bypass')
    assert model.identifier == 'chresmax_v3_lambda.ip_3_chresmax_v3_gpu_8_cl_0.01_ip_3_322_322_18432_c1[_6*3*1_]_bypass'
