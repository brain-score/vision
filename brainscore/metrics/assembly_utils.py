import xarray as xr
from brainio.assemblies import DataAssembly
from brainio.transform import subset


# get one slice of the assembly, only containing the required dims
def _get_slice(asm, dims):
    assert set(dims) <= set(asm.dims)
    isel = {dim: slice(0,1) for dim in asm.dims if dim not in dims}  # using slice(0,1) instead of 0 to make squeeze-drop works!
    return asm.isel(isel).squeeze(drop=True)

# return a dict[dim -> coord DataArray]
def get_coords_except_dims(asm, dims=()):
    return {dim: asm.coords[dim] for dim in asm.dims if dim not in dims}

def get_coords_of_dims(asm, dims=()):
    return {dim: asm.coords[dim] for dim in asm.dims if dim in dims}

# Given coord DataArray, return dict[coord name -> coord value DataArray]
def get_levels(coord_val):
    if coord_val.variable.level_names:
        # if not None
        return {var: coord_val.variable.get_level_variable(var) for var in coord_val.variable.level_names}
    else:
        return {coord_val.variable.name: coord_val.variable}

# return a dict[dim -> dict[coord name -> coord value DataArray]]
def get_coords(asm, dims=None):
    ret = get_coords_of_dims(asm, dims=dims or asm.dims)
    for k, v in ret.items():
        ret[k] = get_levels(v)
    return ret

def replace_coords(asm, coord_dict):
    coords = {}
    for coord_name, coord_val in coord_dict.items():
        if not isinstance(coord_val, dict):
            # is a uni-index
            coords[coord_name] = (coord_name, coord_val)
            continue

        if len(coord_val) == 1:
            # xarray bug: the length-1 multiindex will be squeezed
            var = list(coord_val.keys())[0]
            val = coord_val[var]
            coords[var] = (coord_name, val)
            if var != coord_name:  # if the single coord is not the varname, add both; otherwise sequeeze to uni-index
                coords[coord_name] = (coord_name, val)
        else:
            for var, val in coord_val.items():
                coords[var] = (coord_name, val)
    return DataAssembly(asm.values, dims=asm.dims, coords=coords)


def _is_eql(eql):
    if isinstance(eql, bool):
        return eql
    else:
        return eql.all()

def check_coords_contain(source, target, equal=False):
    # if not equal, only require source <= target in terms of coord names, not require source == target in terms of coord values
    # if equal, all coords must be the same
    for k in source.coords.keys():
        if not (k in target.coords.keys()): 
            return False 

        if hasattr(target[k].variable, "level_names") and target[k].variable.level_names is not None:
            # a multiindex -- we have to check every variable
            for var in target[k].variable.level_names:
                if not (var in source[k].variable.level_names):
                    return False 
                eql = target[var].values == source[var].values
                if not _is_eql(eql):
                    return False 

            if equal:
                for var in source[k].variable.level_names:
                    if not (var in target[k].variable.level_names):
                        return False 
        else:
            eql = target[k].values == source[k].values
            if not _is_eql(eql):
                return False 

    if equal:
        for k in target.coords.keys():
            if not (k in source.coords.keys()):
                return False 

    return True


# subset all assemblies so that they have the exactly same coords along the assigned dims (sorted) 
# and sorted according to the first, transpose them to the front. return view, instead of copy
# if dims_must_equal, the sorted dims must be exactly the same
# dims_must_match == True, the sorted dims of the first assembly must be contained in the following assemblies, so that we can align
def align_dims(*assemblies, dims, dims_must_equal=False):
    asms = assemblies
    dims_must_match = True

    if len(asms) == 0:
        raise ValueError("Input assemblies must not be empty.")
    elif len(asms) == 1:
        pass
    else:
        # subset(s, t): when dims_must_match -> the s must strictly contain t
        tmp = _get_slice(asms[0], dims)
        asms = [subset(asm, tmp, subset_dims=dims, dims_must_match=dims_must_match) for asm in asms]

        # check matching
        if dims_must_equal:
            for asm in asms[1:]:
                slic = _get_slice(asm, dims)
                assert check_coords_contain(tmp, slic, equal=True), "The coords of assemblies do not match in the specified dimensions."

    # align
    coords = get_coords(asms[0], dims=dims)
    coords_to_sort = []
    for c in coords.keys():
        if isinstance(coords[c], dict):
            coords_to_sort.extend(coords[c].keys())
        else:
            coords_to_sort.append(c)
    
    asms = [asm.transpose(*dims, ...).sortby(coords_to_sort) for asm in asms]

    return asms


def apply_over_dims(callable, *asms, dims):
    asms = [asm.transpose(*dims, ...) for asm in asms]
    sizes = [asms[0].sizes[dim] for dim in dims]

    def apply_helper(sizes, dims, *asms):
        xarr = []
        attrs = {}
        size = sizes[0]
        rsizes = sizes[1:]
        dim = dims[0]
        rdims = dims[1:]
        for s in range(size):
            if len(sizes) == 1:
                arr = callable(*[asm.isel({dim:s}) for asm in asms])
            else:
                arr = apply_helper(rsizes, rdims, *[asm.isel({dim:s}) for asm in asms])

            if arr is not None:
                for k,v in arr.attrs.items():
                    assert isinstance(v, xr.DataArray)
                    attrs.setdefault(k, []).append(v.expand_dims(dim))
                xarr.append(arr)
       
        if not xarr:
            return
        else:
            xarr = xr.concat(xarr, dim=dim)
            attrs = {k: xr.concat(vs, dim=dim) for k,vs in attrs.items()}
            xarr.coords[dim] = asms[0].coords[dim]
            for k,v in attrs.items():
                attrs[k].coords[dim] = asms[0].coords[dim]
                xarr.attrs[k] = attrs[k]
            return xarr

    return apply_helper(sizes, dims, *asms)



if __name__ == "__main__":

    ## align

    x = DataAssembly(
        [
            [1, 2],
            [3, 4],
        ],
        dims=["a", "b"],
        coords={"a_0": ("a", [0, 1]), "a_1": ("a", [0, 1]), "b": [0, 1]}
    )

    y = DataAssembly(
        [
            [5, 7],
            [6, 8]
        ],
        dims=["b", "a"],
        coords={"a": [0, 2], "b": [0, 1]}
    ) 

    z = DataAssembly(
        [
            [[2], [4], [99]],
            [[1], [3], [100]],
        ],
        dims=["b", "a", "c"],
        coords={"a_0": ("a", [0, 1, 2]), "a_1": ("a", [0, 1, 2]), "b": [1, 0], "c": [10]}
    )

    d = DataAssembly(
        [
            [[2], [4]],
            [[1], [3]],
        ],
        dims=["d", "a", "c"],
        coords={"a_0": ("a", [0, 1]), "a_1": ("a", [0, 1]), "d": [1, 0], "c": [10]}
    )

    def _except(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)
        else:
            assert False

    _except(align_dims, x, y, dims=["a", "b"])
    x_, y_ = align_dims(x, y, dims=["b"])
    assert (x.transpose("b", ...).sortby(["b"]) == x_).all()
    assert (y==y_).all()
    x_, z_ = align_dims(x, z, dims=["a", "b"])
    assert (x.sortby(["a", "b"]) == x_).all()
    assert (x.sortby(["a", "b"]).values == z_.isel(c=0).values).all()
    _except(align_dims, x, d, dims=["a", "b"])
    _except(align_dims, x, z, dims=["a", "b"], dims_must_equal=True)

    ## compute_over_dim

    from functools import partial
    check_coords_equal = partial(check_coords_contain, equal=True)

    assert check_coords_equal(x, x.transpose("b", "a"))
    assert not check_coords_equal(x, y)
    assert not check_coords_equal(x, z.sortby("b"))

    import numpy as np
    arr = DataAssembly([[1, 2], [3, 4]], dims=["x", "y"], coords={"x_ab": ("x", ["a", "b"]), "x_01": ("x", [0, 1]), "y": [0, 1]})
    def func(ele):
        ret = ele ** 2
        ret.attrs["origin"] = ele
        return ret

    ret = apply_over_dims(func, arr, dims=["x", "y"])
    assert (ret.values == np.array([[1, 4], [9, 16]])).all()
    assert (ret.attrs["origin"].values == arr.values).all()
    assert check_coords_equal(ret, arr)

    ret = apply_over_dims(func, arr, dims=["x"])
    assert (ret.values == np.array([[1, 4], [9, 16]])).all()
    assert (ret.attrs["origin"].values == arr.values).all()
    assert check_coords_equal(ret, arr)
