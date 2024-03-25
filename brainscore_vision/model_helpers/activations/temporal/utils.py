import numpy as np

from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align


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


def stack_with_nan_padding_(arr_list, axis=0, dtype=np.float16):
    # Get shapes of all arrays
    shapes = [np.array(arr.shape) for arr in arr_list]
    max_shape = np.max(shapes, axis=0)

    # Allocate concatenated array with NaN padding
    result = np.full(np.concatenate(([len(arr_list)], max_shape)), np.nan, dtype=dtype)

    # Fill in individual arrays
    for i, arr in enumerate(arr_list):
        slices = tuple(slice(0, s) for s in arr.shape)
        result[i][slices] = arr

    result = np.swapaxes(result, 0, axis)

    return result


def stack_with_nan_padding(arr_list, axis=0, dtype=np.float16):
    # Get shapes of all arrays
    shapes = [np.array(arr.shape) for arr in arr_list]
    max_shape = np.max(shapes, axis=0)

    results = []

    # Fill in individual arrays
    for arr in arr_list:
        result = np.pad(arr, [(0, max_shape[j]-s) for j, s in enumerate(arr.shape)], 
                        mode='constant', constant_values=np.nan)
        results.append(result)

    result = np.stack(results, axis=axis)
    result = np.swapaxes(result, 0, axis)

    return result


def batch_2d_resize(arr, size, mode):
    import cv2
    # arr [N, H, W, C]
    N, H, W, C = arr.shape
    arr = arr.transpose(1, 2, 3, 0).reshape(H, W, C*N)
    if mode == "bilinear":
        mode = cv2.INTER_LINEAR
        ret = cv2_resize(arr, size, mode)
    elif mode == "pool":
        mode = cv2.INTER_AREA
        ret = cv2_resize(arr, size, mode)
    ret = ret.reshape(size[1], size[0], C, N).transpose(3, 0, 1, 2)
    return ret

    
# cv2 has the wierd bug of cannot handling too large channel size
def cv2_resize(arr, size, mode, batch_size=256):
    # arr [H, W, C]
    import cv2
    ori_dtype = arr.dtype
    arr = arr.astype(float)
    C = arr.shape[-1]
    ret = []
    for i in range(0, C, batch_size):
        ret.append(cv2.resize(arr[..., i:i+batch_size], size, interpolation=mode))
    return np.concatenate(ret, axis=-1).astype(ori_dtype)


def parallelize(func, iterable, n_jobs=1, verbose=0, mode="multiprocess"):
    if mode == "multiprocess":
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(item) for item in iterable)
    elif mode == "threading":
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            return list(executor.map(func, iterable))
    else:
        raise ValueError(f"Unknown mode {mode}")


def assembly_align_to_fps(output_assembly, fps, mode="portion"):
    EPS = 1e-9  # prevent the duration from being slightly larger than the last time bin
    interval = 1000 / fps
    duration = output_assembly["time_bin_end"].values[-1]
    target_time_bin_starts = np.arange(0, duration-EPS, interval)
    target_time_bin_ends = target_time_bin_starts + interval
    target_time_bin_ends[-1] = duration  # use this to avoid numerical error
    target_time_bins = [(start, end) for start, end in zip(target_time_bin_starts, target_time_bin_ends)]
    return assembly_time_align(output_assembly, target_time_bins, mode=mode)


# get the inferencer from any model
def get_inferencer(any_model):
    from brainscore_vision.model_helpers.activations.temporal.core import Inferencer, ActivationsExtractor
    from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

    if isinstance(any_model, Inferencer): return any_model
    if isinstance(any_model, ActivationWrapper): return any_model._extractor.inferencer
    if isinstance(any_model, ModelCommitment): return any_model.activations_model._extractor.inferencer
    if isinstance(any_model, ActivationsExtractor): return any_model.inferencer
    raise ValueError(f"Cannot find inferencer from the model {any_model}")

def get_base_model(any_model):
    from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

    if isinstance(any_model, ActivationWrapper): return any_model
    if isinstance(any_model, ModelCommitment): return any_model.activations_model
    raise ValueError(f"Cannot find inferencer from the model {any_model}")


# get the layers from the layer_activation_format
def get_specified_layers(any_model):
    inferencer = get_inferencer(any_model)
    return list(inferencer.layer_activation_format.keys())

# switch the inferencer at any level
# specify key='same' to retrive the same parameter from the original inferencer
def switch_inferencer(any_model, new_inferencer_cls, **kwargs):
    inferencer = get_inferencer(any_model)
    base_model = get_base_model(any_model)
    for k, v in kwargs.items():
        if v == 'same': kwargs[k] = getattr(inferencer, k)
    base_model.build_extractor(new_inferencer_cls, **kwargs)