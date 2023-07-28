import os
from pathlib import Path
import json

import numpy as np
import xarray as xr
import pandas as pd

from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_base.stimuli import StimulusSet
from brainio_collection.packaging import package_data_assembly, package_stimulus_set
from mkgu_packaging.dicarlo.sanghavi import filter_neuroids


def collect_stimuli(data_dir):
    image_dir = data_dir / 'images' / 'things-2'
    assert os.path.isdir(image_dir)
    files = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    files = files[:-130]  # Discard last 130 images (5 grey and 25x5 normalizer images)

    assert os.path.isdir(data_dir / 'image-metadata')
    stimuli = pd.read_csv(data_dir / 'image-metadata' / 'things_2_metadata.csv')

    stimuli = stimuli.rename(columns={'id': 'image_id'})

    stimuli['image_current_local_file_path'] = stimuli.apply(
        lambda row: os.path.join(image_dir, str(row.image_id) + '.jpg'), axis=1)

    assert len(np.unique(stimuli['image_id'])) == len(stimuli)
    stimuli = StimulusSet(stimuli)
    stimuli.image_paths = \
        {stimuli.at[idx, 'image_id']: stimuli.at[idx, 'image_current_local_file_path'] for idx in range(len(stimuli))}
    return stimuli


def load_responses(data_dir, stimuli):
    data_dir = data_dir / 'database'
    assert os.path.isdir(data_dir)
    psth = np.load(data_dir / 'solo.rsvp.things-2.experiment_psth.npy')  # Shaped images x repetitions x time_bins x channels

    # Compute firing rate for given time bins
    timebins = [[70, 170], [170, 270], [50, 100], [100, 150], [150, 200], [200, 250], [70, 270]]
    photodiode_delay = 30  # Delay recorded on photodiode is ~30ms
    timebase = np.arange(-100, 381, 10)  # PSTH from -100ms to 380ms relative to stimulus onset
    assert len(timebase) == psth.shape[2]
    rate = np.empty((len(timebins), psth.shape[0], psth.shape[1], psth.shape[3]))
    for idx, tb in enumerate(timebins):
        t_cols = np.where((timebase >= (tb[0] + photodiode_delay)) & (timebase < (tb[1] + photodiode_delay)))[0]
        rate[idx] = np.mean(psth[:, :, t_cols, :], axis=2)  # Shaped time bins x images x repetitions x channels

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
    psth = np.load(data_dir / 'solo.rsvp.things-2.normalizer_psth.npy')
    t_cols = np.where((timebase >= (70 + photodiode_delay)) & (timebase < (170 + photodiode_delay)))[0]
    rate = np.mean(psth[:, :, t_cols, :], axis=2)
    normalizer_assembly = xr.DataArray(rate,
                                       coords={'repetition': ('repetition', list(range(rate.shape[1]))),
                                               'image_id': ('image', list(range(rate.shape[0]))),
                                               'id': ('image', list(range(rate.shape[0])))},
                                       dims=['image', 'repetition', 'neuroid'])
    for column_name, column_data in neuroid_meta.iteritems():
        normalizer_assembly = normalizer_assembly.assign_coords(
            **{f'{column_name}': ('neuroid', list(column_data.values))})
    normalizer_assembly = normalizer_assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    normalizer_assembly = normalizer_assembly.drop('image')
    normalizer_assembly = normalizer_assembly.transpose('presentation', 'neuroid')
    normalizer_assembly = NeuronRecordingAssembly(normalizer_assembly)

    filtered_assembly = filter_neuroids(normalizer_assembly, 0.7)
    assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    # Add other experiment and data processing related info
    assembly.attrs['image_size_degree'] = 8
    assembly.attrs['stim_on_time_ms'] = 100

    return assembly


def main():
    data_dir = Path(__file__).parents[6] / 'data2' / 'active' / 'users' / 'sachis'
    assert os.path.isdir(data_dir)

    stimuli = collect_stimuli(data_dir)
    stimuli.identifier = 'dicarlo.THINGS2'
    assembly = load_responses(data_dir, stimuli)
    assembly.name = 'dicarlo.SanghaviMurty2020THINGS2'

    print('Packaging stimuli')
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')
    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio.dicarlo')
    return


if __name__ == '__main__':
    main()
