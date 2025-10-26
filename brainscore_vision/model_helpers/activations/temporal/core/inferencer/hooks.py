from brainscore_vision.model_helpers.activations.temporal.utils import batch_2d_resize

def _make_tensor_to_numpy_hook():
    def hook(val, layer, stimulus):
        return val.cpu().data.numpy()
    return hook

def _make_dtype_hook(dtype):
    return lambda val, layer, stimulus: val.astype(dtype)

# downsample the activations with the largest spatial size (among width and height) to the specified size
def _make_spatial_downsample_hook(max_spatial_size, layer_activation_format, mode="pool"):
    def hook(val, layer, stimulus):
        if max_spatial_size is None:
            return val

        dims = layer_activation_format[layer]

        # require both H and W dimensions to do spatial downsampling
        if "H" not in dims or "W" not in dims:
            return val

        H_dim, W_dim = dims.index("H"), dims.index("W")
        val = val.swapaxes(H_dim, 0).swapaxes(W_dim, 1)
        shape = val.shape[2:]
        h, w = val.shape[:2]
        val = val.reshape(h, w, -1)
        new_size = _compute_new_size(w, h, max_spatial_size)
        new_val = batch_2d_resize(val[None,:], new_size, mode=mode)[0]
        new_val = new_val.reshape(*new_size, *shape)
        new_val = new_val.swapaxes(0, H_dim).swapaxes(1, W_dim)
        return new_val
    return hook

def _compute_new_size(w, h, max_spatial_size):
    if isinstance(max_spatial_size, int):
        if h > w:
            new_h = max_spatial_size
            new_w = int(w * new_h / h)
        else:
            new_w = max_spatial_size
            new_h = int(h * new_w / w)
    else:
        new_h = int(h * max_spatial_size)
        new_w = int(w * max_spatial_size)
    
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    return new_h, new_w
