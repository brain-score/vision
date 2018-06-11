from collections import OrderedDict

from mkgu.assemblies import walk_coords


def collect_coords(assembly, ignore_dims, rename_coords_list, kind):
    coords = assembly.coords
    filtered_coords = filter_coords(coords, ignore_dims=ignore_dims)
    renamed_coords = rename_coords(filtered_coords, rename_coords_list=rename_coords_list, suffix=kind)
    return renamed_coords


def filter_coords(coords, ignore_dims):
    def filter_func(coord_values):
        coord, values = coord_values
        if coord in ignore_dims:
            return False
        value_dims_ignore = [dim in ignore_dims for dim in values.dims]
        if any(value_dims_ignore):
            assert all(value_dims_ignore)
            return False
        return True

    return dict(filter(filter_func, coords.items()))


def rename_coords(coords, rename_coords_list, suffix):
    coord_names = {coord: coord if coord not in rename_coords_list else coord + '-' + suffix for coord in coords}
    return {coord_names[name]: (tuple(coord_names[dim] for dim in values.dims), values.values)
            for name, values in coords.items()}


def collect_dim_shapes(assembly, rename_dims_list, ignore_dims, kind):
    dims = assembly.dims
    filtered_dims = filter_dims(dims, ignore_dims=ignore_dims)
    renamed_dims = rename_dims(filtered_dims, rename_dims_list=rename_dims_list, rename_suffix=kind)
    shapes = OrderedDict((dim, assembly[dim].shape) for dim in filtered_dims)
    renamed = OrderedDict((renamed_dim, shape) for renamed_dim, (_, shape) in zip(renamed_dims, shapes.items()))
    return renamed


def filter_dims(dims, ignore_dims):
    return [dim for dim in dims if dim not in ignore_dims]


def rename_dims(dims, rename_dims_list, rename_suffix):
    return [dim if dim not in rename_dims_list else dim + '-' + rename_suffix for dim in dims]


def get_modified_coords(assembly, modifier=lambda name, dims, values: (name, (dims, values))):
    coords = {}
    for name, dims, values in walk_coords(assembly):
        name_dims_vals = modifier(name, dims, values)
        if name_dims_vals is not None:
            name, (dims, vals) = name_dims_vals
            coords[name] = dims, vals
    return coords


def merge_dicts(dicts):
    result = {}
    for dict in dicts:
        result = {**result, **dict}
    return result
