import os
import re
import zipfile
from glob import glob

import brainscore_vision
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from brainscore_vision.knownfile import KnownFile as kf
from brainscore_vision.lookup import pwdb
from brainscore_vision.assemblies import AssemblyModel, AssemblyStoreMap, AssemblyStoreModel
from brainscore_vision.stimuli import ImageModel, AttributeModel, ImageMetaModel, StimulusSetModel, ImageStoreModel, \
    StimulusSetImageMap, ImageStoreMap

# from FreemanZiemba2013_V1V2data_readme.m
textureNumOrder = [327, 336, 393, 402, 13, 18, 23, 30, 38, 48, 52, 56, 60, 71, 99]


def load_stimuli(stimuli_directory):
    stimuli = []
    for image_file_path in glob(f"{stimuli_directory}/*.png"):
        image_file_name = os.path.basename(image_file_path)
        fields = fields_from_image_name(image_file_name)
        im_kf = kf(image_file_path)
        extra_fields = {
            'image_file_path': image_file_path,
            'image_file_name': image_file_name,
            "image_file_sha1": im_kf.sha1,
            "image_id": im_kf.sha1,
            "image_store_path": "movshon_stimuli/" + image_file_name
        }
        stimuli.append({**fields, **extra_fields})


    stimuli = pd.DataFrame(stimuli)
    assert len(stimuli) == 15 * 2 * 15
    assert len(np.unique(stimuli['texture_family'])) == 15
    assert len(np.unique(stimuli['texture_type'])) == 2
    assert len(np.unique(stimuli['sample'])) == 15
    assert len(np.unique(stimuli["image_id"])) == len(stimuli)

    return stimuli


def load_responses(response_file, stimuli_directory):
    # from the readme.m: data is in the form:
    # (texFamily) x (texType) x (sample) x (rep) x (timeBin) x (cellNum)
    # (15)        x (2)       x (15)     x (20)  x (300)     x (102+)
    # in python, ordering is inverted:
    # (cellNum) x (timeBin) x (rep) x (sample) x (texType) x (texFamily)
    # (102+)    x (300)     x (20)  x (15)     x (2)       x (15)
    responses = h5py.File(response_file, 'r')
    v1, v2 = responses['v1'], responses['v2']
    assert v1.shape[1:] == v2.shape[1:]  # same except cells
    responses = np.concatenate([v1, v2])

    assembly = xr.DataArray(responses,
                            coords={
                                'neuroid_id': ("neuroid", list(range(1, responses.shape[0] + 1))),
                                'region': ('neuroid', ['V1'] * v1.shape[0] + ['V2'] * v2.shape[0]),
                                'time_bin_start': ("time_bin", list(range(responses.shape[1]))),  # each bin is 1 ms
                                'time_bin_end': ("time_bin", list(range(1, responses.shape[1] + 1))),
                                'repetition': list(range(responses.shape[2])),
                                'sample': list(range(1, responses.shape[3] + 1)),
                                'texture_type': ["noise", "texture"],
                                'texture_family': textureNumOrder
                            },
                            dims=['neuroid', 'time_bin', 'repetition', 'sample', 'texture_type', 'texture_family'])

    assembly = assembly.stack(presentation=['texture_type', 'texture_family', 'sample', 'repetition'])

    image_fields = zip(*[assembly[k].values for k in ['texture_type', 'texture_family', 'sample']])
    image_names = [image_name_from_fields(im[0], "320x320", im[1], im[2]) for im in image_fields]
    assembly["image_file_name"] = ("presentation", image_names)

    kfs = {}
    sha1s = []
    for image_name in image_names:
        if image_name in kfs:
            im_kf = kfs[image_name]
        else:
            im_kf = kf(os.path.join(stimuli_directory, image_name))
            kfs[image_name] = im_kf
        sha1s.append(im_kf.sha1)
    assembly["image_id"] = ("presentation", sha1s)

    return brainscore_vision.assemblies.NeuronRecordingAssembly(assembly)


def write_netcdf(assembly, target_netcdf_file):
    assembly.reset_index(assembly.indexes.keys(), inplace=True)
    result = assembly.drop(["image_file_name", "texture_type", "texture_family", "sample"])
    result.reset_index(result.indexes.keys(), inplace=True)
    result.to_netcdf(target_netcdf_file)


def create_image_zip(stimuli, target_zip_path):
    os.makedirs(os.path.dirname(target_zip_path), exist_ok=True)
    with zipfile.ZipFile(target_zip_path, 'w') as target_zip:
        for image in stimuli.itertuples():
            target_zip.write(image.image_file_path, arcname=image.image_store_path)
    zip_kf = kf(target_zip_path)
    return zip_kf.sha1


def add_image_lookup(stimuli, target_zip_path, zip_sha1, stimulus_set_name, image_store_unique_name, bucket_name):
    pwdb.connect(reuse_if_open=True)
    zip_file_name = os.path.basename(target_zip_path)

    stim_set_model, created = StimulusSetModel.get_or_create(name=stimulus_set_name)
    image_store, created = ImageStoreModel.get_or_create(location_type="S3", store_type="zip",
                                                         location=f"https://{bucket_name}.s3.amazonaws.com/{zip_file_name}",
                                                         unique_name=image_store_unique_name,
                                                         sha1=zip_sha1)
    eav_image_file_sha1, created = AttributeModel.get_or_create(name="image_file_sha1", type="str")
    eav_image_file_name, created = AttributeModel.get_or_create(name="image_file_name", type="str")
    eav_image_texture_type, created = AttributeModel.get_or_create(name="texture_type", type="str")
    eav_image_texture_family, created = AttributeModel.get_or_create(name="texture_family", type="int")
    eav_image_sample, created = AttributeModel.get_or_create(name="sample", type="int")
    eav_image_resolution, created = AttributeModel.get_or_create(name="resolution", type="str")

    for image in stimuli.itertuples():
        pw_image, created = ImageModel.get_or_create(image_id=image.image_id)
        pw_stimulus_set_image_map, created = StimulusSetImageMap.get_or_create(stimulus_set=stim_set_model, image=pw_image)
        pw_image_image_store_map, created = ImageStoreMap.get_or_create(image=pw_image, image_store=image_store,
                                                                        path=image.image_store_path)
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_file_sha1, value=str(image.image_file_sha1))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_file_name, value=str(image.image_file_name))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_texture_type, value=str(image.texture_type))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_texture_family, value=str(image.texture_family))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_sample, value=str(image.sample))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_resolution, value=str(image.resolution))

    return stim_set_model


def add_assembly_lookup(assembly_name, stim_set_model, bucket_name, target_netcdf_file, assembly_store_unique_name):
    kf_netcdf = kf(target_netcdf_file)
    assy, created = AssemblyModel.get_or_create(name=assembly_name, assembly_class="NeuronRecordingAssembly",
                                                stimulus_set=stim_set_model)
    store, created = AssemblyStoreModel.get_or_create(assembly_type="netCDF",
                                                      location_type="S3",
                                                      location=f"https://{bucket_name}.s3.amazonaws.com/{assembly_name}.nc",
                                                      unique_name=assembly_store_unique_name,
                                                      sha1=kf_netcdf.sha1)
    assy_store_map, created = AssemblyStoreMap.get_or_create(assembly_model=assy, assembly_store_model=store, role=assembly_name)


def image_name_from_fields(texture_type, resolution, texture_family, sample):
    mapping = {"noise": "noise", "texture": "tex"}
    return f"{mapping[texture_type]}-{resolution}-im{int(texture_family)}-smp{int(sample)}.png"


def fields_from_image_name(image_name):
    # sample image file name: noise-320x320-im13-smp8
    integer_fields = ['family', 'sample']
    mapping = {"noise": "noise", "tex": "texture"}
    pattern = "^(?P<texture_type>[^-]+)-(?P<resolution>[^-]+)-im(?P<texture_family>[0-9]*)-smp(?P<sample>[0-9]+)\.png$"
    match = re.match(pattern, image_name)
    assert match
    fields = match.groupdict()
    fields = {field: value if field not in integer_fields else int(value) for field, value in fields.items()}
    fields = {field: value if field != "texture_type" else mapping[value] for field, value in fields.items()}
    return fields


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'FreemanZiemba2013')
    stimuli_directory = os.path.join(data_path, 'stim')
    response_file = os.path.join(data_path, 'data', 'FreemanZiemba2013_V1V2data.mat')
    output_path = os.path.join(data_path, 'out')
    stimulus_set_name = "FreemanZiemba2013"
    bucket_name = "brain-score-movshon"
    image_store_unique_name = "image_movshon_stimuli"
    target_zip_path = os.path.join(output_path, image_store_unique_name + ".zip")
    assembly_name = stimulus_set_name
    assembly_store_unique_name = "assy_movshon_FreemanZiemba2013"
    target_netcdf_file = os.path.join(output_path, assembly_name + ".nc")

    stimuli = load_stimuli(stimuli_directory)
    assembly = load_responses(response_file, stimuli_directory)

    nonzero = np.count_nonzero(assembly)
    assert nonzero > 0

    all_ids = lambda assembly, stimuli, i: assembly.sel(image_file_name=stimuli["image_file_name"][i])["image_id"]
    all_match = lambda assembly, stimuli, i: all(all_ids(assembly, stimuli, i) == stimuli["image_id"][i])
    assert all([all_match(assembly, stimuli, i) for i in range(len(stimuli))])

    zip_sha1 = create_image_zip(stimuli, target_zip_path)
    stim_set_model = add_image_lookup(stimuli, target_zip_path, zip_sha1, stimulus_set_name, image_store_unique_name, bucket_name)
    write_netcdf(assembly, target_netcdf_file)
    add_assembly_lookup(assembly_name, stim_set_model, bucket_name, target_netcdf_file, assembly_store_unique_name)

    return (assembly, stimuli)


if __name__ == '__main__':
    main()
