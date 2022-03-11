import os

import h5py
import numpy as np
import pandas as pd
import imageio
import xarray as xr
from pathlib import Path
from result_caching import store
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set, package_data_assembly


@store(identifier_ignore=['stimuli_dir'])
def collect_stimuli(data_path, stimuli_dir):
    stimuli, local_paths = [], []
    with h5py.File(data_path, 'r') as f:
        meta = f['meta']
        os.makedirs(stimuli_dir, exist_ok=True)
        for image_id, filename, reference, obj, sample_number, object_number, phase in tqdm(zip(
                *[meta[key][0] for key in ['id', 'filename', 'img', 'obj', 'sampleNr', 'objNr', 'phase']]),
                desc='stimuli', total=len(meta['img'][0])):
            # gather meta
            image_id = ''.join([chr(i[0]) for i in f[image_id]])
            object_name = ''.join(chr(i[0]) for i in f[f[obj][0][0]])
            sample_number = int(f[sample_number][0][0])
            object_number = int(f[object_number][0][0])
            filename = ''.join([chr(i[0]) for i in f[filename]])
            stimuli.append({
                'image_path_within_store': filename,
                'image_id': image_id,
                'object_name': object_name,
                'obj': object_name,
                'image_label': object_name,
                'sample_number': sample_number,
                'object_number': object_number,
            })
            # write image locally (will be packaged later)
            img = np.array(f[reference])
            img = img.transpose([1, 2, 0])
            target_path = os.path.join(stimuli_dir, filename)
            imageio.imwrite(target_path, np.rot90(np.fliplr(img)))
            local_paths.append(target_path)
    stimuli = StimulusSet(stimuli)
    stimuli.image_paths = {image_id: local_path for image_id, local_path in zip(stimuli['image_id'], local_paths)}
    return stimuli


@store(identifier_ignore=['stimuli'])
def collect_data(data_path, stimuli):
    responses = []
    with h5py.File(data_path, 'r') as f:
        data = f['data']
        keys = ['sample_number', 'sample_object_number', 'distractor_number', 'correct']
        for trial in tqdm(np.array(data['dataTurk'], dtype=int).transpose(), desc='trials'):
            response = {key: value for key, value in zip(keys, trial)}
            meta = stimuli[stimuli['sample_number'] == response['sample_number']]
            assert len(meta) == 1
            meta = meta.iloc[0]
            response['sample_obj'] = response['sample_object'] = response['truth'] = meta['obj']
            response['image_id'] = meta['image_id']
            distractor_meta = stimuli[stimuli['object_number'] == response['distractor_number']]
            distractor_obj = set(distractor_meta['obj'])
            assert len(distractor_obj) == 1
            response['dist_obj'] = response['distractor_object'] = list(distractor_obj)[0]
            response['correct'] = bool(response['correct'])
            response['choice'] = response['sample_obj'] if response['correct'] else response['dist_obj']
            del response['correct']  # cannot be serialized to netcdf
            responses.append(response)

    responses = pd.DataFrame(responses)
    responses = to_xarray(responses)
    return responses


def to_xarray(responses):
    columns = list(responses.columns)
    responses = xr.DataArray(responses['choice'],
                             coords={column: ('presentation', responses[column]) for column in columns},
                             dims=['presentation'])
    responses = responses.set_index(presentation=columns)
    responses = BehavioralAssembly(responses)
    return responses


def main():
    data_dir = Path('/braintree/home/msch/share/kar2018_cocobehavior')
    data_path = data_dir / 'coco_data.mat'
    stimuli = collect_stimuli(data_path, stimuli_dir=data_dir / 'stimuli')
    assembly = collect_data(data_path, stimuli)
    assert len(stimuli) == 1600
    assert len(assembly) == 179660
    assert len(set(assembly['image_id'].values)) == 1600
    assert set(assembly['image_id'].values) == set(stimuli['image_id'])
    assert len(set(assembly['choice'].values)) == len(set(assembly['sample_obj'].values)) \
           == len(set(assembly['dist_obj'].values)) == 10

    assembly.name = 'dicarlo.Kar2018coco_behavior'
    stimuli.name = 'dicarlo.Kar2018coco_color'

    # split into public/private
    # Rajalingham2018 has 585_511 public and 341_785 private trials, with 2_160 unique public and 240 private images.
    # We here trade-off between the number of images for fitting and the number of trials for testing
    # by settling on a 80:20 split.
    print("Splitting into public/private")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=15)
    public_idx, private_idx = next(split.split(stimuli['image_id'].values, stimuli['obj'].values))
    public_stimuli, private_stimuli = stimuli.iloc[public_idx], stimuli.iloc[private_idx]
    assert len(public_stimuli) == 1_280 and len(private_stimuli) == 320
    public_stimuli.name, private_stimuli.name = stimuli.name + '.public', stimuli.name + '.private'
    public_assembly = assembly[{'presentation': [image_id in public_stimuli['image_id'].values
                                                 for image_id in assembly['image_id'].values]}]
    private_assembly = assembly[{'presentation': [image_id in private_stimuli['image_id'].values
                                                  for image_id in assembly['image_id'].values]}]
    assert len(public_assembly) == 143_492 and len(private_assembly) == 36_168
    public_assembly.name, private_assembly.name = assembly.name + '.public', assembly.name + '.private'

    # package
    return
    print("Packaging")
    for stimuli, assembly in zip([public_stimuli, private_stimuli], [public_assembly, private_assembly]):
        package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.name,
                             bucket_name="brainio.dicarlo")
        package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.name,
                              assembly_class='BehavioralAssembly', bucket_name="brainio.dicarlo")


if __name__ == '__main__':
    main()
