import numpy as np
from brainscore_vision.model_helpers.activations.temporal.utils import (
    batch_2d_resize
)

def test_proportional_average_pooling():
    arr = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])[...,None].astype(float)

    size_0 = (2, 1)
    arr_0 = np.array([
        [3, 4],
    ])[...,None]

    def proportional_average_pooling(arr, size):
        return batch_2d_resize(arr[None,:], size, "pool")[0]

    assert np.allclose(proportional_average_pooling(arr, size_0), arr_0)

    size_1 = (2, 2)
    arr_1 = np.array([
        [1*2/3+3*1/3, 2*2/3+4*1/3],
        [3*1/3+5*2/3, 4*1/3+6*2/3],
    ])[...,None]

    assert np.allclose(proportional_average_pooling(arr, size_1), arr_1)

    size_2 = (1, 2)
    arr_2 = np.array([
        [(1*2/3+3*1/3+2*2/3+4*1/3)/2],
        [(3*1/3+5*2/3+4*1/3+6*2/3)/2],
    ])[...,None]

    assert np.allclose(proportional_average_pooling(arr, size_2), arr_2)

    size_3 = (3, 3)
    arr_3 = np.array([
        [1, 1.5, 2],
        [3, 3.5, 4],
        [5, 5.5, 6],
    ])[...,None]

    assert np.allclose(proportional_average_pooling(arr, size_3), arr_3)

    size_4 = (1, 1)
    arr_4 = arr.reshape(-1, 1).mean(0)[None, None, :]
    assert np.allclose(proportional_average_pooling(arr, size_4), arr_4)

    size_5 = (2, 4)
    arr_5 = np.array([
        [1, 2],
        [(1*1+3*2)/3, (2*1+4*2)/3],
        [(3*2+5*1)/3, (4*2+6*1)/3],
        [5, 6],
    ])[...,None].astype(float)
    assert np.allclose(proportional_average_pooling(arr, size_5), arr_5)
