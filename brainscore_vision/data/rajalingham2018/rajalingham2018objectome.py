import os
import zipfile
from glob import glob
from pathlib import Path

import boto3
import pandas as pd
import xarray as xr

import mkgu_packaging
from brainio_base.assemblies import BehavioralAssembly
from brainio_base.stimuli import StimulusSet
from brainio_collection.lookup import pwdb
from brainio_collection.knownfile import KnownFile as kf
from brainio_collection.assemblies import AssemblyModel, AssemblyStoreMap, AssemblyStoreModel
from brainio_collection.stimuli import ImageModel, AttributeModel, ImageMetaModel, StimulusSetModel, ImageStoreModel, \
    StimulusSetImageMap, ImageStoreMap


def get_objectome(source_data_path):
    objectome = pd.read_pickle(os.path.join(source_data_path, 'objectome24s100_humanpool.pkl'))
    # objectome['correct'] = objectome['choice'] == objectome['sample_obj']
    objectome['truth'] = objectome['sample_obj']

    subsample = pd.read_pickle(os.path.join(source_data_path, 'objectome24s100_imgsubsampled240_pandas.pkl'))
    objectome['enough_human_data'] = objectome['id'].isin(subsample.values[:, 0])
    objectome = to_xarray(objectome)
    return objectome


def to_xarray(objectome):
    columns = objectome.columns
    objectome = xr.DataArray(objectome['choice'],
                             coords={column: ('presentation', objectome[column]) for column in columns},
                             dims=['presentation'])
    objectome = objectome.rename({'id': 'image_id'})
    objectome = objectome.set_index(presentation=[col if col != 'id' else 'image_id' for col in columns])
    objectome = BehavioralAssembly(objectome)
    return objectome


def load_stimuli(meta_assembly, source_stim_path):
    stimuli_paths = list(glob(os.path.join(source_stim_path, '*.png')))
    stimuli_paths.sort()
    stimuli = StimulusSet({'image_current_local_file_path': stimuli_paths,
                           'image_id': [os.path.splitext(os.path.basename(filepath))[0] for filepath in stimuli_paths],
                           'image_path_within_store': [os.path.basename(filepath) for filepath in stimuli_paths]})

    assert all(meta_assembly['sample_obj'].values == meta_assembly['truth'].values)
    image_meta = {image_id: coord_value for image_id, coord_value in
                  zip(meta_assembly['image_id'].values, meta_assembly['sample_obj'].values)}
    meta_values = [image_meta[image_id] for image_id in stimuli['image_id'].values]
    stimuli['image_sample_obj'] = meta_values
    stimuli['image_label'] = stimuli['image_sample_obj']
    return stimuli


def load_responses(source_data_path):
    objectome = get_objectome(source_data_path)
    fitting_objectome, testing_objectome = objectome.sel(enough_human_data=False), objectome.sel(enough_human_data=True)
    return objectome, fitting_objectome, testing_objectome


def create_image_zip(stimuli, target_zip_path):
    os.makedirs(os.path.dirname(target_zip_path), exist_ok=True)
    with zipfile.ZipFile(target_zip_path, 'w') as target_zip:
        for image in stimuli.itertuples():
            target_zip.write(image.image_current_local_file_path, arcname=image.image_path_within_store)
    zip_kf = kf(target_zip_path)
    return zip_kf.sha1


def write_netcdf(assembly, target_netcdf_file):
    assembly.reset_index(assembly.indexes.keys(), inplace=True)
    assembly.to_netcdf(target_netcdf_file)


def add_stimulus_set_metadata_and_lookup_to_db(stimuli, stimulus_set_name, bucket_name, zip_file_name,
                                               image_store_unique_name, zip_sha1):
    pwdb.connect(reuse_if_open=True)
    stim_set_model, created = StimulusSetModel.get_or_create(name=stimulus_set_name)
    image_store, created = ImageStoreModel.get_or_create(location_type="S3", store_type="zip",
                                                         location=f"https://{bucket_name}.s3.amazonaws.com/{zip_file_name}",
                                                         unique_name=image_store_unique_name,
                                                         sha1=zip_sha1)
    add_image_metadata_to_db(stimuli, stim_set_model, image_store)
    return stim_set_model


def add_image_metadata_to_db(stimuli, stim_set_model, image_store):
    pwdb.connect(reuse_if_open=True)
    eav_image_sample_obj, created = AttributeModel.get_or_create(name="image_sample_obj", type="str")
    eav_image_label, created = AttributeModel.get_or_create(name="image_label", type="str")

    for image in stimuli.itertuples():
        pw_image, created = ImageModel.get_or_create(image_id=image.image_id)
        pw_stimulus_set_image_map, created = StimulusSetImageMap.get_or_create(stimulus_set=stim_set_model, image=pw_image)
        pw_image_image_store_map, created = ImageStoreMap.get_or_create(image=pw_image, image_store=image_store,
                                                                        path=image.image_path_within_store)
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_sample_obj, value=str(image.image_sample_obj))
        ImageMetaModel.get_or_create(image=pw_image, attribute=eav_image_label, value=str(image.image_label))


def add_assembly_lookup(assembly_name, stim_set_model, bucket_name, target_netcdf_file, assembly_store_unique_name):
    kf_netcdf = kf(target_netcdf_file)
    assy, created = AssemblyModel.get_or_create(name=assembly_name, assembly_class="BehavioralAssembly",
                                                stimulus_set=stim_set_model)
    store, created = AssemblyStoreModel.get_or_create(assembly_type="netCDF",
                                                      location_type="S3",
                                                      location=f"https://{bucket_name}.s3.amazonaws.com/{assembly_store_unique_name }.nc",
                                                      unique_name=assembly_store_unique_name,
                                                      sha1=kf_netcdf.sha1)
    assy_store_map, created = AssemblyStoreMap.get_or_create(assembly_model=assy, assembly_store_model=store, role=assembly_name)


def upload_to_s3(source_file_path, bucket_name, target_s3_key):
    client = boto3.client('s3')
    client.upload_file(source_file_path, bucket_name, target_s3_key)


def main():
    pkg_path = Path(mkgu_packaging.__file__).parent
    source_path = Path("/braintree/home/msch/share/objectome")
    source_data_path = source_path / 'data'
    source_stim_path = source_path / 'stim'
    target_path = pkg_path.parent / "objectome" / "out"
    target_bucket_name = "brainio-dicarlo"
    assembly_name = "dicarlo.Rajalingham2018"

    public_stimulus_set_unique_name = "objectome.public"
    public_image_store_unique_name = "stimulus_objectome_public"
    public_assembly_unique_name = "dicarlo.Rajalingham2018.public"
    public_assembly_store_unique_name = "assy_dicarlo_Rajalingham2018_public"
    public_target_zip_basename = public_image_store_unique_name + ".zip"
    public_target_zip_path = target_path / public_target_zip_basename
    public_target_netcdf_basename = public_assembly_store_unique_name + ".nc"
    public_target_netcdf_path = target_path / public_target_netcdf_basename
    public_target_zip_s3_key = public_target_zip_basename
    public_target_netcdf_s3_key = public_target_netcdf_basename

    private_stimulus_set_unique_name = "objectome.private"
    private_image_store_unique_name = "stimulus_objectome_private"
    private_assembly_unique_name = "dicarlo.Rajalingham2018.private"
    private_assembly_store_unique_name = "assy_dicarlo_Rajalingham2018_private"
    private_target_zip_basename = private_image_store_unique_name + ".zip"
    private_target_zip_path = target_path / private_target_zip_basename
    private_target_netcdf_basename = private_assembly_store_unique_name + ".nc"
    private_target_netcdf_path = target_path / private_target_netcdf_basename
    private_target_zip_s3_key = private_target_zip_basename
    private_target_netcdf_s3_key = private_target_netcdf_basename

    [all_assembly, public_assembly, private_assembly] = load_responses(source_data_path)
    all_assembly.name = assembly_name
    public_assembly.name = public_assembly_unique_name
    private_assembly.name = private_assembly_unique_name
    all_stimuli = load_stimuli(all_assembly, source_stim_path)
    public_stimuli = all_stimuli[all_stimuli['image_id'].isin(public_assembly['image_id'].values)]
    private_stimuli = all_stimuli[all_stimuli['image_id'].isin(private_assembly['image_id'].values)]
    public_stimuli.name = public_stimulus_set_unique_name
    private_stimuli.name = private_stimulus_set_unique_name

    assert len(public_assembly) + len(private_assembly) == len(all_assembly) == 927296
    assert len(private_assembly) == 341785
    assert len(set(public_assembly['image_id'].values)) == len(public_stimuli) == 2160
    assert len(set(private_assembly['image_id'].values)) == len(private_stimuli) == 240
    assert set(all_stimuli['image_id'].values) == set(all_assembly['image_id'].values)
    assert set(public_stimuli['image_id'].values) == set(public_assembly['image_id'].values)
    assert set(private_stimuli['image_id'].values) == set(private_assembly['image_id'].values)
    assert len(set(private_assembly['choice'].values)) == len(set(public_assembly['choice'].values)) == 24

    print([assembly.name for assembly in [all_assembly, public_assembly, private_assembly]])

    public_zip_sha1 = create_image_zip(public_stimuli, public_target_zip_path)
    public_stimulus_set_model = add_stimulus_set_metadata_and_lookup_to_db(public_stimuli, public_stimulus_set_unique_name, target_bucket_name, public_target_zip_basename, public_image_store_unique_name, public_zip_sha1)
    write_netcdf(public_assembly, public_target_netcdf_path)
    add_assembly_lookup(public_assembly_unique_name,public_stimulus_set_model,target_bucket_name,public_target_netcdf_path, public_assembly_store_unique_name)

    private_zip_sha1 = create_image_zip(private_stimuli, private_target_zip_path)
    private_stimulus_set_model = add_stimulus_set_metadata_and_lookup_to_db(private_stimuli, private_stimulus_set_unique_name, target_bucket_name, private_target_zip_basename, private_image_store_unique_name, private_zip_sha1)
    write_netcdf(private_assembly, private_target_netcdf_path)
    add_assembly_lookup(private_assembly_unique_name,private_stimulus_set_model,target_bucket_name,private_target_netcdf_path, private_assembly_store_unique_name)

    print("uploading to S3")
    upload_to_s3(str(public_target_zip_path), target_bucket_name, public_target_zip_s3_key)
    upload_to_s3(str(public_target_netcdf_path), target_bucket_name, public_target_netcdf_s3_key)
    upload_to_s3(str(private_target_zip_path), target_bucket_name, private_target_zip_s3_key)
    upload_to_s3(str(private_target_netcdf_path), target_bucket_name, private_target_netcdf_s3_key)

    return [(public_assembly, public_stimuli), (private_assembly, private_stimuli)]


if __name__ == '__main__':
    main()
