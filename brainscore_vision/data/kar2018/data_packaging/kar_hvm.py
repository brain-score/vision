import h5py
import numpy as np
import xarray as xr
from pathlib import Path

import brainio_collection
from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_contrib.packaging import package_data_assembly
from mkgu_packaging.dicarlo.kar2018 import filter_neuroids


def load_responses(response_file, additional_coords):
    responses = h5py.File(response_file, 'r')
    assemblies = []
    neuroid_id_offset = 0
    for monkey in responses.keys():
        spike_rates = responses[monkey]['rates']
        assembly = xr.DataArray(spike_rates.value,
                                coords={**{
                                    'image_num': ('image_id', list(range(spike_rates.shape[0]))),
                                    'neuroid_id': ('neuroid', list(
                                        range(neuroid_id_offset, neuroid_id_offset + spike_rates.shape[1]))),
                                    'region': ('neuroid', ['IT'] * spike_rates.shape[1]),
                                    'monkey': ('neuroid', [monkey] * spike_rates.shape[1]),
                                    'repetition': list(range(spike_rates.shape[2])),
                                }, **additional_coords},
                                dims=['image_id', 'neuroid', 'repetition'])
        assemblies.append(assembly)
        neuroid_id_offset += spike_rates.shape[1]
    assembly = xr.concat(assemblies, 'neuroid')
    assembly = assembly.stack(presentation=['image_id', 'repetition'])
    assembly = NeuronRecordingAssembly(assembly)
    assert len(assembly['presentation']) == 640 * 63
    assert len(np.unique(assembly['image_id'])) == 640
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


def load_stimuli_ids(data_dir):
    # these stimuli_ids are SHA1 hashes on generative parameters
    stimuli_ids = h5py.File(data_dir / 'hvm640_ids.mat', 'r')
    stimuli_ids = [''.join(chr(c) for c in stimuli_ids[stimuli_ids['hvm640_ids'].value[0, i]])
                   for i in range(stimuli_ids['hvm640_ids'].value[0].size)]
    # we use the filenames to reference into our packaged StimulusSet ids
    stimuli_filenames = h5py.File(data_dir / 'hvm640_names.mat', 'r')
    stimuli_filenames = [''.join(chr(c) for c in stimuli_filenames[stimuli_filenames['hvm640_img_names'].value[0, i]])
                         for i in range(stimuli_filenames['hvm640_img_names'].value[0].size)]
    # the stimuli_ids in our packaged StimulusSets are SHA1 hashes on pixels.
    # we thus need to reference between those two ids.
    packaged_stimuli = brainio_collection.get_stimulus_set('dicarlo.hvm')
    reference_table = {row.image_file_name: row.image_id for row in packaged_stimuli.itertuples()}
    referenced_ids = [reference_table[filename] for filename in stimuli_filenames]
    return {'image_id': ('image_id', referenced_ids),
            'image_generative_id': ('image_id', stimuli_ids)}


def main():
    data_dir = Path(__file__).parent / 'hvm'
    stimuli_ids = load_stimuli_ids(data_dir)

    assembly = load_responses(data_dir / 'hvm640_neural.h5', additional_coords=stimuli_ids)
    assembly.name = 'dicarlo.Kar2018hvm'

    package_data_assembly(assembly, data_assembly_name=assembly.name, stimulus_set_name='dicarlo.hvm',
                          bucket_name='brainio-dicarlo')


if __name__ == '__main__':
    main()
