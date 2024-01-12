import os
from pathlib import Path
import json

import numpy as np
import xarray as xr
import pandas as pd
import scipy.io as sio

from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_base.stimuli import StimulusSet
from brainio_collection.packaging import package_data_assembly, package_stimulus_set
from mkgu_packaging.dicarlo.sanghavi import filter_neuroids


pd.set_option('display.max_columns', None)


def collect_stimuli(data_dir):
    data = sio.loadmat(data_dir / 'image-metadata' / 'objectome_metadata.mat', squeeze_me=True)
    column_names = [x for x in list(data.keys()) if not x.startswith('__')]
    stimuli = pd.DataFrame()
    for col in column_names:
        stimuli[col] = data[col]
    stimuli = stimuli.rename(columns={'id': 'image_id', 'obj': 'image_label'})
    stimuli = stimuli.sort_values(by='image_id').reset_index(drop=True)
    print(stimuli)


    # data_dir = data_dir / 'images' / 'nat300'
    # assert os.path.isdir(data_dir)
    # files = sorted(os.listdir(data_dir), key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    # stimuli['image_id'] = ''
    # stimuli['image_file_name'] = ''
    # stimuli['image_current_local_file_path'] = ''
    # stimuli['image_path_within_store'] = ''
    # stimuli['image_file_sha1'] = ''
    # stimuli['id'] = ''
    #
    # for idx, image_file_name in enumerate(files):
    #     image_file_path = os.path.join(data_dir, image_file_name)
    #     im_kf = kf(image_file_path)
    #
    #     stimuli.at[idx, 'image_id'] = im_kf.sha1
    #     stimuli.at[idx, 'image_file_name'] = image_file_name
    #     stimuli.at[idx, 'image_current_local_file_path'] = image_file_path
    #     stimuli.at[idx, 'image_path_within_store'] = image_file_name
    #     stimuli.at[idx, 'image_file_sha1'] = im_kf.sha1
    #     stimuli.at[idx, 'id'] = int(os.path.splitext(image_file_name)[0].split('_')[-1])
    #
    # assert len(np.unique(stimuli['image_id'])) == len(stimuli)
    # stimuli = StimulusSet(stimuli)
    # stimuli.image_paths = \
    #     {stimuli.at[idx, 'image_id']: stimuli.at[idx, 'image_current_local_file_path'] for idx in range(len(stimuli))}
    return stimuli


def load_responses(data_dir, stimuli):
    data_dir = data_dir / 'database'
    assert os.path.isdir(data_dir)
    psth = np.load(data_dir / 'solo.rsvp.nat300.experiment_psth.npy')  # Shaped images x repetitions x time_bins x channels

    # Compute firing rate for given time bins
    timebins = [[70, 170], [50, 100], [100, 150], [150, 200], [200, 250], [70, 270]]
    photodiode_delay = 30  # Delay recorded on photodiode is ~30ms
    timebase = np.arange(-100, 381, 10)  # PSTH from -100ms to 380ms relative to stimulus onset
    assert len(timebase) == psth.shape[2]
    rate = np.empty((len(timebins), psth.shape[0], psth.shape[1], psth.shape[3]))
    for idx, tb in enumerate(timebins):
        t_cols = np.where((timebase >= (tb[0] + photodiode_delay)) & (timebase < (tb[1] + photodiode_delay)))[0]
        rate[idx] = np.mean(psth[:, :, t_cols, :], axis=2)  # Shaped images x repetitions x channels

    assembly = xr.DataArray(rate,
                            coords={'repetition': ('repetition', list(range(rate.shape[2]))),
                                    'time_bin_id': ('time_bin', list(range(rate.shape[0]))),
                                    'time_bin_start': ('time_bin', [x[0] for x in timebins]),
                                    'time_bin_stop': ('time_bin', [x[1] for x in timebins])},
                            dims=['time_bin', 'image', 'repetition', 'neuroid'])

    # Add neuroid related meta data
    neuroid_meta = pd.DataFrame(json.load(open(data_dir.parent / 'array-metadata' / 'mapping.json')))
    for column_name, column_data in neuroid_meta.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    # Add stimulus related meta data
    for column_name, column_data in stimuli.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('image', list(column_data.values))})

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = assembly.drop('image')
    assembly = NeuronRecordingAssembly(assembly)

    # Filter noisy electrodes
    filtered_assembly = filter_neuroids(assembly.sel(time_bin_id=0).squeeze(dim='time_bin'), 0.7)
    assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    # Add other experiment and data processing related info
    assembly.attrs['image_size_degree'] = 8
    assembly.attrs['stim_on_time_ms'] = 100

    return assembly


def main():
    data_dir = Path(__file__).parents[6] / 'data2' / 'active' / 'users' / 'sachis'
    assert os.path.isdir(data_dir)

    import brainio_collection
    print(brainio_collection.list_stimulus_sets())
    print(brainio_collection.list_assemblies())
    stimuli = brainio_collection.get_stimulus_set('dicarlo.objectome.public').sort_values(by='image_id')
    print(stimuli)
    # print(stimuli.columns)
    stimuli = collect_stimuli(data_dir)
    # stimuli.name = 'dicarlo.Rust2012'
    # assembly = load_responses(data_dir, stimuli)
    # assembly.name = 'dicarlo.SanghaviKar2020'

    # print('Packaging stimuli')
    # package_stimulus_set(stimuli, stimulus_set_name=stimuli.name, bucket_name='brainio-dicarlo')
    # print('Packaging assembly')
    # package_data_assembly(assembly, data_assembly_name=assembly.name, stimulus_set_name=stimuli.name,
    #                       bucket_name='brainio-dicarlo')
    return


if __name__ == '__main__':
    main()
