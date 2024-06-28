import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import re
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_collection.knownfile import KnownFile as kf
from brainio_contrib.packaging import package_stimulus_set, package_data_assembly
from mkgu_packaging.dicarlo.kar2018 import filter_neuroids


def collect_stimuli(stimuli_directory):
    meta = os.path.join(stimuli_directory, 'cocogray_labels.mat')
    meta = h5py.File(meta, 'r')
    labels = [''.join(chr(c) for c in meta[meta['lb'].value[0, i]]) for i in range(meta['lb'].value[0].size)]
    stimuli = []
    for image_file_path in tqdm(glob(os.path.join(stimuli_directory, '*.png'))):
        image_file_name = os.path.basename(image_file_path)
        image_number = re.match('im([0-9]+).png', image_file_name)
        image_number = int(image_number.group(1))
        im_kf = kf(image_file_path)
        stimuli.append({
            'image_id': im_kf.sha1,
            'image_file_name': image_file_name,
            'image_current_local_file_path': image_file_path,
            'image_file_sha1': im_kf.sha1,
            'image_number': image_number,
            'image_path_within_store': image_file_name,
            'label': labels[image_number],
        })
    stimuli = pd.DataFrame(stimuli)
    assert len(stimuli) == 1600
    assert len(np.unique(stimuli['image_id'])) == len(stimuli)
    return stimuli


def load_responses(response_file, stimuli):
    responses = h5py.File(response_file, 'r')
    assemblies = []
    neuroid_id_offset = 0
    for monkey in responses.keys():
        spike_rates = responses[monkey]['rates']
        assembly = xr.DataArray(spike_rates.value,
                                coords={
                                    'image_num': ('image_id', list(range(spike_rates.shape[0]))),
                                    'image_id': ('image_id', [
                                        stimuli['image_id'][stimuli['image_number'] == num].values[0]
                                        for num in range(spike_rates.shape[0])]),
                                    'neuroid_id': ('neuroid', list(
                                        range(neuroid_id_offset, neuroid_id_offset + spike_rates.shape[1]))),
                                    'region': ('neuroid', ['IT'] * spike_rates.shape[1]),
                                    'monkey': ('neuroid', [monkey] * spike_rates.shape[1]),
                                    'repetition': list(range(spike_rates.shape[2])),
                                },
                                dims=['image_id', 'neuroid', 'repetition'])
        assemblies.append(assembly)
        neuroid_id_offset += spike_rates.shape[1]
    assembly = xr.concat(assemblies, 'neuroid')
    assembly = assembly.stack(presentation=['image_id', 'repetition'])
    assembly = NeuronRecordingAssembly(assembly)
    assert len(assembly['presentation']) == 1600 * 45
    assert len(np.unique(assembly['image_id'])) == 1600
    assert len(assembly.sel(monkey='nano')['neuroid']) == len(assembly.sel(monkey='magneto')['neuroid']) == 288
    assert len(assembly['neuroid']) == len(np.unique(assembly['neuroid_id'])) == 288 * 2
    # filter noisy electrodes
    assembly = filter_neuroids(assembly, threshold=.7)
    # add time info
    assembly = assembly.expand_dims('time_bin')
    assembly['time_bin_start'] = 'time_bin', [70]
    assembly['time_bin_end'] = 'time_bin', [170]
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
    return assembly


def main():
    data_dir = Path(__file__).parent / 'coco'
    stimuli = collect_stimuli(data_dir / 'stimuli')
    stimuli.name = 'Kar2018cocogray'
    assembly = load_responses(data_dir / 'cocoGray_neural.h5', stimuli)
    assembly.name = 'dicarlo.Kar2018cocogray'

    print("Packaging stimuli")
    package_stimulus_set(stimuli, stimulus_set_name=stimuli.name,
                         bucket_name="brainio-dicarlo")
    print("Packaging assembly")
    package_data_assembly(assembly, data_assembly_name=assembly.name, stimulus_set_name=stimuli.name,
                          bucket_name="brainio-dicarlo")


if __name__ == '__main__':
    main()
