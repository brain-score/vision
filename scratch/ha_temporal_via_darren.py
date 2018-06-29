
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import os
import tables
import scipy.io as sio
import matplotlib.pyplot as plt
import brainscore


def coords_from_df(dim, df, name_map):
    coords_d = {}
    for col in name_map:
        col_ser = df[name_map[col]]
        if col_ser.dtype.kind in ["S", "O"]:
            col_ser = col_ser.astype("unicode")
        coords_d[col] = (dim, col_ser)
    return coords_d


def parse_mat_name(mat_name):
    spl = mat_name.split("_")

    if spl[0].endswith("R"):
        hemisphere = "R"
        animal = spl[0][:-1]
    else:
        hemisphere = "L"
        animal = spl[0]

    region = spl[1]

    if len(spl) == 3:
        arr = "P"
    else:
        arr = spl[2]

    return {"hemisphere": hemisphere, "animal": animal, "region": region, "array": arr}


def map_file_from_mat_name(map_dir, mat_name):
    mat_d = parse_mat_name(mat_name)
    basename = mat_d["animal"] + "_" + mat_d["array"] + "_map.mat"
    return os.path.join(map_dir, basename)


def run_whole_notebook(for_jonas_dir, for_jjpr_dir):
    for_jonas_dir_ls = [os.path.join(for_jonas_dir, x) for x in os.listdir(for_jonas_dir)]

    for_jjpr_dir_ls = [os.path.join(for_jjpr_dir, x) for x in os.listdir(for_jjpr_dir)]

    mats = {os.path.basename(mat_file): sio.loadmat(mat_file) for mat_file in for_jonas_dir_ls}

    mat_name = "Tito_V4_HVM6.mat"
    mat = mats[mat_name]

    hvm_temporal_subset = build_assy(mat, mat_name, for_jjpr_dir)

    hvm_temporal_subset_filename = "../data/hvm_temporal_" + mat_name[:-4] + ".nc"

    save_netcdf(hvm_temporal_subset, hvm_temporal_subset_filename)

    # ### testing neuroid coords

    # sum across time bin, mean across reps, leaves neuroids and images, treat that as a vector of length <num of images>
    # for each neuroid

    # compare to same neuroid ID from hvm assy, for same image set, images in same order
    # get aligned pair, hvm_temporal_aligned and hvm_aligned
    # for context, correlate each neuroid vector in hvm_temporal_aligned with each neuroid vector in hvm_aligned

    hvm_temporal_subset_collapsed = hvm_temporal_subset.sum("time_bin")
    hvm_temporal_subset_collapsed = hvm_temporal_subset_collapsed.multi_groupby(presentation_groups).mean(
        "presentation")
    hvm_temporal_subset_collapsed = coords_reset(hvm_temporal_subset_collapsed, "neuroid", "neuroid_id")

    hvm = brainscore.get_assembly("HvM")

    hvm_collapsed = hvm.multi_groupby(["image_id", "image_file_name"]).mean("presentation").squeeze()
    hvm_collapsed = coords_reset(hvm_collapsed, "neuroid", "neuroid_id")

    hvm_temporal_aligned, hvm_aligned = xr.align(hvm_temporal_subset_collapsed, hvm_collapsed, join="inner")

    # does not protect from div by zero
    def normalize_axis_1(arr):
        return arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]

    hvm_temporal_normed = normalize_axis_1(hvm_temporal_aligned)

    hvm_normed = normalize_axis_1(hvm_aligned)

    each_to_self = np.array([np.correlate(m, n)[0] for m, n in zip(hvm_temporal_normed, hvm_normed)])

    # compare each neuroid vector in hvm_temporal_normed with each neuroid vector in hvm_normed
    each_to_each = np.array([[np.correlate(m, n)[0] for m in hvm_temporal_normed] for n in hvm_normed])

    show_me(lambda m, n: np.correlate(m, n)[0], hvm_temporal_normed, hvm_normed)

    show_me(lambda m, n: np.correlate(m, n)[0], hvm_temporal_aligned, hvm_aligned)

    show_me(lambda m, n: np.correlate(m, n)[0], hvm_temporal_subset_collapsed, hvm_collapsed)

    show_me(lambda m, n: np.linalg.norm(m - n), hvm_temporal_normed, hvm_normed)

    def collapse_1_assy_temporal(assy_temporal):
        assy_temporal_collapsed = assy_temporal.sum("time_bin")
        assy_temporal_collapsed = assy_temporal_collapsed.multi_groupby(presentation_groups).mean("presentation")
        assy_temporal_collapsed = coords_reset(assy_temporal_collapsed, "neuroid", "neuroid_id")
        return assy_temporal_collapsed

    def show_me_mat(func, assy_temporal_collapsed):
        aligned = xr.align(assy_temporal_collapsed, hvm_collapsed, join="inner")
        show_me(func, *aligned)

    # corr = lambda m, n: np.correlate(m, n)[0]
    # mats["Tito_IT_M_HVM6.mat"]
    # show_me_mat(corr, collapse_1_assy_temporal())


# func to flatten MultiIndexes
# has to be inplace because of DataAssembly's constructor bug;  when that's fixed, inplace isn't necessary
def flatten_multiindexes(xr_data):
    temp = xr_data.copy()
    temp.reset_index(all_index_levels(xr_data), inplace=True)
    return temp


# get all the index levels on a dimension except for one keeper
def coords_drops(da, dim, keeper):
    return [x for x in da.indexes[dim].names if x != keeper]


# remove all index levels on a dim except for one keeper
def coords_reset(da, dim, keeper):
    drops = coords_drops(da, dim, keeper)
    return da.reset_index(drops, drop=True).set_index(keeper)


def make_assy_file_name(data_dir, da_name):
    basename = "hvm_temporal_" + da_name + ".nc"
    return os.path.join(data_dir, basename)


# func to get index level names
def levels_for_index(xr_data, index):
    return xr_data.indexes[index].names


def all_index_levels(xr_data):
    nested = [levels_for_index(xr_data, index) for index in xr_data.indexes]
    return [x for inner in nested for x in inner]


def save_netcdf(assy, filename):
    flatten_multiindexes(assy).to_netcdf(filename)


def build_assys(mats, for_jjpr_dir):
    assys = {}
    for mat_name in mats:
        mat = mats[mat_name]
        if "orig_elecs" in mat:
            assys[mat_name[:-4]] = build_assy(mat, mat_name, for_jjpr_dir)
    return assys


def write_assys_to_disk(assys, data_dir):
    for da_name in assys:
        hvm_temporal_subset = assys[da_name]
        hvm_temporal_subset_filename = make_assy_file_name(data_dir, da_name)
        save_netcdf(hvm_temporal_subset, hvm_temporal_subset_filename)


def load_assy_from_disk(assy_file_name):
    return brainscore.assemblies.NeuronRecordingAssembly(xr.open_dataarray(assy_file_name))


presentation_groups = ["image_id", "image_file_name"]


def show_me(func, first, second):
    each_to_each = np.array([[func(m, n) for m in first] for n in second])
    plt.imshow(each_to_each, interpolation='nearest')
    plt.show()


def build_assy(mat, mat_name, map_dir):
    dims_original = ["images", "reps", "time", "units"]  # per Darren Seibert, email 2017-10-17
    dims = ["stimulus", "repetition", "time_bin", "neuroid"]

    da_scratch = build_data_array(dims, mat, mat_name, map_dir)

    # ### Construct DataAssembly
    stacked = da_scratch.stack(presentation=("repetition", "stimulus"))
    transposed = stacked.transpose("neuroid", "presentation", "time_bin")
    transposed.reset_index("presentation", inplace=True)
    hvm_temporal_subset = brainscore.assemblies.NeuronRecordingAssembly(transposed)
    return hvm_temporal_subset


def build_data_array(dims, mat, mat_name, map_dir):
    coords_stimulus = get_coords_stimulus(mat)
    coords_time_bin = get_coords_time_bin()
    coords_neuroid = get_coords_neuroid(mat, mat_name, map_dir)
    coords_repetition = get_coords_repetition(mat)

    # ### Constructing a DataArray
    coords = {}
    coords.update(coords_stimulus)
    coords.update(coords_repetition)
    coords.update(coords_time_bin)
    coords.update(coords_neuroid)
    da_scratch = xr.DataArray(mat["bins"], dims=dims, coords=coords)
    return da_scratch


def get_coords_repetition(mat):
    coords_repetition = {
        "repetition_index": ("repetition", range(mat["bins"].shape[1]))
    }
    return coords_repetition


def get_coords_neuroid(mat, mat_name, map_dir):
    # ### Building coords for neuroid
    len_neuroid = mat["bins"].shape[3]
    coords_neuroid_uniform = parse_mat_name(mat_name)
    df_neuroid = pd.DataFrame(
        {field: [coords_neuroid_uniform[field]] * len_neuroid for field in coords_neuroid_uniform})
    df_neuroid["original_electrodes"] = mat["orig_elecs"].squeeze()
    # find the .cmp or .txt or excel or .mat file corresponding to Tito_V4_HVM6.mat
    map_file_name = map_file_from_mat_name(map_dir, mat_name)
    electrode_map = sio.loadmat(map_file_name)
    df_electrode_map = pd.DataFrame({"row": electrode_map["row"].squeeze(), "column": electrode_map["col"].squeeze()})
    joined_orig_elecs = pd.merge(df_neuroid, df_electrode_map,
                                 how="left", left_on="original_electrodes", right_index=True)
    id_build = joined_orig_elecs["animal"] + ["_"] * len_neuroid
    id_build = id_build + joined_orig_elecs["hemisphere"] + ["_"] * len_neuroid
    id_build = id_build + joined_orig_elecs["array"] + ["_"] * len_neuroid
    id_build = id_build + joined_orig_elecs["row"].astype("unicode") + ["_"] * len_neuroid
    id_build = id_build + joined_orig_elecs["column"].astype("unicode")
    joined_orig_elecs["neuroid_id"] = id_build
    neuroid_map = {x: x for x in joined_orig_elecs.columns}
    coords_neuroid = coords_from_df("neuroid", joined_orig_elecs, neuroid_map)
    return coords_neuroid


def get_coords_time_bin():
    # ### Building coords for time bin
    time_bin_start = np.arange(0, 500, 10)
    coords_time_bin = {
        "time_bin_start": ("time_bin", time_bin_start),
        "time_bin_end": ("time_bin", time_bin_start + 10),
        "time_bin_center": ("time_bin", time_bin_start + 5),
    }
    return coords_time_bin


def get_coords_stimulus(mat):
    # ### Building coords for stimulus
    image_df_cols = {
        "image_index": range(len(mat["stm_file_names"])),
        "image_file_name": np.core.defchararray.strip(mat["stm_file_names"])
    }
    df_stm_file_names = pd.DataFrame(image_df_cols)
    simpler_meta_file = "/braintree/home/qbilius/.streams/hvm/meta.pkl"
    df_simpler_meta = pd.read_pickle(simpler_meta_file)
    joined_simpler = pd.merge(df_stm_file_names, df_simpler_meta, how="left", left_on="image_file_name",
                              right_on="filename").sort_values("image_index")

    stimulus_map = {
        'image_background_id': 'bg_id',
        'category_name': 'category',
        'image_file_name': 'image_file_name',
        'image_axis_index': "image_index",
        'image_id': 'id',
        'object_name': 'objname',
        'rxy': 'rxy',
        'rxy_semantic': 'rxy_semantic',
        'rxz': 'rxz',
        'rxz_semantic': 'rxz_semantic',
        'ryz': 'ryz',
        'ryz_semantic': 'ryz_semantic',
        's': 's',
        'image_size': 'size',
        'ty': 'ty',
        'tz': 'tz',
        'variation': 'var'
    }

    coords_stimulus = coords_from_df("stimulus", joined_simpler, stimulus_map)
    return coords_stimulus


def scratch_build_assy(for_jonas_dir, for_jjpr_dir):
    for_jonas_dir_ls = [os.path.join(for_jonas_dir, x) for x in os.listdir(for_jonas_dir)]

    for_jjpr_dir_ls = [os.path.join(for_jjpr_dir, x) for x in os.listdir(for_jjpr_dir)]

    mats = {os.path.basename(mat_file): sio.loadmat(mat_file) for mat_file in for_jonas_dir_ls}

    mat_name = "Tito_V4_HVM6.mat"
    mat = mats[mat_name]

    hvm_temporal_subset = build_assy(mat, mat_name, for_jjpr_dir)

    print(hvm_temporal_subset)


def scratch_test_neuroid_ids():
    data_dir = "/braintree/home/jjpr/dev/scratch/mkgu_scratch/data"
    da_name = "Chabo_IT_A_HVM6"
    assy_file_name = make_assy_file_name(data_dir, da_name)
    assy = load_assy_from_disk(assy_file_name)
    print(assy)


def main():
    for_jonas_dir = "/braintree/data2/active/common/for_jonas"

    for_jjpr_dir = "/braintree/data2/active/common/for_jjpr"

    # scratch_build_assy(for_jonas_dir, for_jjpr_dir)
    scratch_test_neuroid_ids()
    # run_whole_notebook(for_jonas_dir, for_jjpr_dir)
    print("Current directory:  " + os.getcwd())
    pass


if __name__ == '__main__':
    main()

