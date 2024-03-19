import numpy as np


def concat_with_nan_padding(arr_list, axis=0, dtype=np.float16):
    # Get shapes of all arrays
    shapes = [np.array(arr.shape) for arr in arr_list]
    max_shape = np.max(shapes, axis=0)
    len_axis = sum([arr.shape[axis] for arr in arr_list])

    # Allocate concatenated array with NaN padding
    result_shape = [s if i != axis else len_axis for i, s in enumerate(max_shape)]
    result = np.full(result_shape, np.nan, dtype=dtype)

    # Fill in individual arrays
    offset = 0
    for arr in arr_list:
        slices = [slice(0, s) if i != axis else slice(offset, offset+s) 
                  for i, s in enumerate(arr.shape)]
        offset += arr.shape[axis]
        result[slices] = arr

    return result


def stack_with_nan_padding(arr_list, axis=0, dtype=np.float16):
    # Get shapes of all arrays
    shapes = [np.array(arr.shape) for arr in arr_list]
    max_shape = np.max(shapes, axis=0)

    # Allocate concatenated array with NaN padding
    result = np.full(np.concatenate(([len(arr_list)], max_shape)), np.nan, dtype=dtype)

    # Fill in individual arrays
    for i, arr in enumerate(arr_list):
        slices = tuple(slice(0, s) for s in arr.shape)
        result[i][slices] = arr

    return result