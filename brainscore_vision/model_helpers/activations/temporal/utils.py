import os
import numpy as np
import pickle

from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align
from brainio.assemblies import DataAssembly


# allow efficient fill_value for memmap
class custom_memmap(np.memmap):
    def __new__(subtype, filename, dtype=np.uint8, mode='r+', offset=0,
                shape=None, order='C', fill_value=None):
        # Import here to minimize 'import numpy' overhead
        import mmap
        import os.path
        import struct
        from numpy import ndarray

        mode_equivalents = {
            "readonly":"r",
            "copyonwrite":"c",
            "readwrite":"r+",
            "write":"w+"
        }

        dtypedescr = np.dtype
        valid_filemodes = ["r", "c", "r+", "w+"]
        writeable_filemodes = ["r+", "w+"]

        try:
            mode = mode_equivalents[mode]
        except KeyError as e:
            if mode not in valid_filemodes:
                raise ValueError(
                    "mode must be one of {!r} (got {!r})"
                    .format(valid_filemodes + list(mode_equivalents.keys()), mode)
                ) from None

        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given if mode == 'w+'")

        def get_ctx(mode):
            if hasattr(filename, 'read'):
                f_ctx = nullcontext(filename)
            else:
                f_ctx = open(
                    os.fspath(filename),
                    ('r' if mode == 'c' else mode)+'b'
                )
            return f_ctx

        with get_ctx(mode) as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize

            if shape is None:
                bytes = flen - offset
                if bytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                            "multiple of the data-type size.")
                size = bytes // _dbytes
                shape = (size,)
            else:
                if type(shape) not in (tuple, list):
                    try:
                        shape = [operator.index(shape)]
                    except TypeError:
                        pass
                shape = tuple(shape)
                size = np.intp(1)  # avoid default choice of np.int_, which might overflow
                for k in shape:
                    size *= k

            bytes = int(offset + size*_dbytes)

            if mode in ('w+', 'r+') and flen < bytes:
                fid.seek(bytes - 1, 0)
                fid.write(b'\0')
                fid.flush()

            if mode == 'w+' and fill_value is not None:
                val = np.array(fill_value).astype(dtype).tobytes()
                TARGET_BLOCK_SIZE = 1024 * 1024 * 10
                # chunk writing
                for i in range(0, bytes, TARGET_BLOCK_SIZE):
                    with get_ctx(mode='r+') as _fid:
                        _fid.seek(i)
                        _fid.write(val * min(TARGET_BLOCK_SIZE, bytes - i))
                        _fid.flush()

            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE

            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            array_offset = offset - start
            mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)

            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode

            if isinstance(filename, os.PathLike):
                # special case - if we were constructed with a pathlib.path,
                # then filename is a path object, not a string
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                # py3 returns int for TemporaryFile().name
                self.filename = os.path.abspath(fid.name)
            # same as memmap copies (e.g. memmap + 1)
            else:
                self.filename = None

        return self


# a map that write directly to the disk without loading into memory
class data_assembly_mmap:
    def __init__(self, filepath=None, **kwargs):
        self.filepath = filepath
        self.kwargs = kwargs
        self._in_memory = filepath is None

        if self._in_memory:
            self._data = np.full(**kwargs)
            self._created = True
        else:
            self._data = None
            self._created = False
            self.data_file = os.path.join(self.filepath, "data.npy")
            self.meta_file = os.path.join(self.filepath, "meta.pkl")

    def _open(self):
        if self._in_memory:
            return

        if self._data is None:
            kwargs = self.kwargs.copy()
            fill_value = self.kwargs.get("fill_value", None)
            if not self._created:
                self._data = custom_memmap(self.data_file, mode='w+', **kwargs)
                self._created = True
            else:
                self._data = custom_memmap(self.data_file, mode='r+', **kwargs)

    def _close(self):
        if self._data is not None:
            self._data.flush()
            del self._data
            self._data = None

    def __setitem__(self, key, value):
        self._open()
        self._data[key] = value
        if not self._in_memory:
            self._close()

    def __getitem__(self, key):
        self._open()
        return self._data[key]

    def register_meta(self, dims, coords):
        self.coords = coords
        self.dims = dims

        if not self._in_memory:
            with open(self.meta_file, 'wb') as f:
                pickle.dump((dims, coords, self.kwargs), f)

    def to_assembly(self):
        self._open()
        return DataAssembly(self._data, coords=self.coords, dims=self.dims)

    @staticmethod
    def is_saved(filepath):
        data_file = os.path.join(filepath, "data.npy")
        meta_file = os.path.join(filepath, "meta.pkl")
        return os.path.exists(data_file) and os.path.exists(meta_file)

    @staticmethod
    def load(filepath):
        if filepath is None:
            return None

        if not data_assembly_mmap.is_saved(filepath):
            return None

        meta_file = os.path.join(filepath, "meta.pkl")
        with open(meta_file, 'rb') as f:
            dims, coords, kwargs = pickle.load(f)

        data = data_assembly_mmap(filepath, **kwargs)
        data._created = True
        data._open()
        data.dims = dims
        data.coords = coords

        return data
        

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


def stack_with_nan_padding(arr_list, axis=0, dtype=None):
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
    if dtype is not None and result.dtype != dtype:
        result = result.astype(dtype)

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
        # ret = proportional_average_pooling(arr, size)
        mode = cv2.INTER_AREA
        ret = cv2_resize(arr, size, mode)
    else:
        raise ValueError(f"Unknown mode {mode}")
    ret = ret.reshape(size[1], size[0], C, N).transpose(3, 0, 1, 2)
    return ret


def proportional_average_pooling(arr, size):
    import cv2
    mode = cv2.INTER_AREA
    ret = cv2_resize(arr, size, mode)
    return ret


def proportional_average_pooling_(arr, size):
    H, W, C = arr.shape
    w, h = size

    ret = np.zeros((h, w, C))
    for r in range(h):
        for c in range(w):
            y0 = r * H / h
            x0 = c * W / w
            y1 = (r + 1) * H / h
            x1 = (c + 1) * W / w
            r_start = int(y0)
            r_end = np.ceil(y1).astype(int)
            c_start = int(x0)
            c_end = np.ceil(x1).astype(int)
            val = arr[r_start:r_end, c_start:c_end]
            y_overlap_start = np.maximum(y0, np.arange(r_start, r_end))
            y_overlap_end = np.minimum(y1, np.arange(r_start, r_end)+1)
            x_overlap_start = np.maximum(x0, np.arange(c_start, c_end))
            x_overlap_end = np.minimum(x1, np.arange(c_start, c_end)+1)
            y_overlap = y_overlap_end - y_overlap_start
            x_overlap = x_overlap_end - x_overlap_start
            areas = np.outer(y_overlap, x_overlap)
            areas = areas / np.sum(areas)
            val = (val * areas[...,None]).reshape(-1, C).sum(0)
            ret[r, c] = val
    return ret.astype(arr.dtype)


# cv2 has the wierd bug of cannot handling too large channel size
def cv2_resize(arr, size, mode, batch_size=4):
    # arr [H, W, C]
    import cv2
    ori_dtype = arr.dtype
    arr = arr.astype(float)
    C = arr.shape[-1]
    ret = []
    for i in range(0, C, batch_size):
        val = cv2.resize(arr[..., i:i+batch_size], size, interpolation=mode)
        if len(val.shape)<3: val = val[...,None]
        ret.append(val)
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


## model utils

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


## Others
    
def download_weight_file(url, folder=None):
    import requests
    from tqdm import tqdm
    weight_fname = os.path.basename(url)
    brainscore_home = os.getenv("BRAINSCORE_HOME", os.path.expanduser("~/.brain-score"))
    model_cache = os.path.join(brainscore_home, "models")
    if folder: model_cache = os.path.join(model_cache, folder)
    os.makedirs(model_cache, exist_ok=True)
    weight_path = os.path.join(model_cache, weight_fname)

    if os.path.exists(weight_path):
        return weight_path

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"{weight_fname}") as progress_bar:
        with open(weight_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")

    return weight_path