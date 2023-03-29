import os

import h5py
import numpy as np
import pandas as pd
import scipy.misc
import scipy.misc
import scipy.misc
import xarray as xr
from result_caching import store
from tqdm import tqdm

from brainio_base.assemblies import DataAssembly
from brainio_base.stimuli import StimulusSet
from brainio_collection.knownfile import KnownFile
from brainio_contrib.packaging import package_stimulus_set, package_data_assembly

object_lookup = {
    1: 'bear',
    2: 'elephant',
    3: 'person',
    4: 'car',
    5: 'dog',
    6: 'apple',
    7: 'chair',
    8: 'plane',
    9: 'bird',
    10: 'zebra',
}


@store(identifier_ignore=['stimuli_dir'])
def collect_stimuli(data_path, stimuli_dir):
    stimuli = []
    with h5py.File(data_path, 'r') as f:
        images, objects = f['images'], f['obj']
        os.makedirs(stimuli_dir, exist_ok=True)
        for image_num, (image, obj) in tqdm(enumerate(zip(images, objects[0])), desc='stimuli', total=len(images)):
            image = image.transpose([1, 2, 0])
            obj = object_lookup[obj]
            filename = f"image_{image_num:04d}.jpg"
            target_path = os.path.join(stimuli_dir, filename)
            scipy.misc.imsave(target_path, np.rot90(np.fliplr(image)))
            im_kf = KnownFile(target_path)
            stimuli.append({
                'image_current_local_file_path': target_path,
                'image_path_within_store': filename,
                "image_file_sha1": im_kf.sha1,
                "image_id": im_kf.sha1,
                'image_num': image_num,
                'image_label': obj,
            })
    stimuli = StimulusSet(stimuli)
    return stimuli


@store(identifier_ignore=['stimuli'])
def collect_data(data_folder, stimuli):
    data = []
    with h5py.File(os.path.join(data_folder, 'dataset.h5'), 'r') as svm_f, \
            h5py.File(os.path.join(data_folder, 'ost_on_logistic.mat'), 'r') as logistic_f:
        svm_osts, logistic_osts, i1s = svm_f['ost'], logistic_f['ost'], svm_f['i1']
        for svm_ost, logistic_ost, i1, stimuli_row in tqdm(zip(
                svm_osts[0], logistic_osts[0], i1s[0], stimuli.itertuples()), total=len(svm_osts), desc='trials'):
            row = {'ost-svm': svm_ost, 'ost-logistic': logistic_ost, 'i1': i1,
                   'image_id': stimuli_row.image_id, 'image_label': stimuli_row.image_label}
            data.append(row)
    data = pd.DataFrame(data)
    data = to_xarray(data)
    return data


def to_xarray(data):
    presentation_columns = [column for column in data.columns if column not in ['ost-svm', 'ost-logistic']]
    data = xr.DataArray(np.stack((data['ost-svm'], data['ost-logistic'])),
                        coords={**{column: ('presentation', data[column]) for column in presentation_columns},
                                **{'decoder': ['svm', 'logistic']}},
                        dims=['decoder', 'presentation'])
    data = data.set_index(presentation=presentation_columns)
    data = DataAssembly(data)
    return data


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'Kar2019OST')
    target_dir = os.path.join(os.path.dirname(__file__), 'Kar2019OST')
    target_stimuli_dir = os.path.join(target_dir, 'stimuli')
    stimuli = collect_stimuli(os.path.join(data_path, 'dataset.h5'), target_stimuli_dir)
    assembly = collect_data(data_path, stimuli)

    assert len(stimuli) == len(assembly['presentation']) == 1320
    assert len(set(assembly['image_id'].values)) == 1318
    assert set(assembly['image_id'].values) == set(stimuli['image_id'])
    assert len(set(assembly['image_label'].values)) == 10

    assembly.name = 'dicarlo.Kar2019'
    stimuli.name = 'dicarlo.Kar2019'
    package(assembly, stimuli)


def package(assembly, stimuli):
    print("Packaging stimuli")
    package_stimulus_set(stimuli, stimulus_set_name=stimuli.name)

    print("Packaging assembly")
    package_data_assembly(assembly, data_assembly_name=assembly.name, stimulus_set_name=stimuli.name)


if __name__ == '__main__':
    main()
