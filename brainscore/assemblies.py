import xarray as xr


def merge_data_arrays(data_arrays):
    # https://stackoverflow.com/a/50125997/2225200
    merged = xr.merge([similarity.rename('z') for similarity in data_arrays])['z'].rename(None)
    # ensure same class
    return type(data_arrays[0])(merged)


def array_is_element(arr, element):
    return len(arr) == 1 and arr[0] == element


def walk_coords(assembly):
    """
    walks through coords and all levels, just like the `__repr__` function, yielding `(name, dims, values)`.
    """
    coords = {}

    for name, values in assembly.coords.items():
        # partly borrowed from xarray.core.formatting#summarize_coord
        is_index = name in assembly.dims
        if is_index and values.variable.level_names:
            for level in values.variable.level_names:
                level_values = assembly.coords[level]
                yield level, level_values.dims, level_values.values
        else:
            yield name, values.dims, values.values
    return coords
