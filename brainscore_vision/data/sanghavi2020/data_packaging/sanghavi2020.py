import os
from pathlib import Path
import json

import numpy as np
import xarray as xr
import pandas as pd

import brainio_collection
from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_collection.packaging import package_data_assembly
from mkgu_packaging.dicarlo.sanghavi import filter_neuroids


def load_responses(data_dir, stimuli):
    psth = np.load(data_dir / 'solo.rsvp.hvm.experiment_psth.npy')  # Shaped images x repetitions x time_bins x channels

    # Drop first (index 0) and second last session (index 25) since they had only one repetition each
    # Actually not, since we're sticking to older protocol re: data cleaning for now
    # psth = np.delete(psth, (0, 25), axis=1)

    # Compute firing rate for given time bins
    timebins = [[70, 170], [170, 270], [50, 100], [100, 150], [150, 200], [200, 250], [70, 270]]
    photodiode_delay = 30  # Delay recorded on photodiode is ~30ms
    timebase = np.arange(-100, 381, 10)  # PSTH from -100ms to 380ms relative to stimulus onset
    assert len(timebase) == psth.shape[2]
    rate = np.empty((len(timebins), psth.shape[0], psth.shape[1], psth.shape[3]))
    for idx, tb in enumerate(timebins):
        t_cols = np.where((timebase >= (tb[0] + photodiode_delay)) & (timebase < (tb[1] + photodiode_delay)))[0]
        rate[idx] = np.mean(psth[:, :, t_cols, :], axis=2)  # Shaped time bins x images x repetitions x channels

    # Load image related meta data (id ordering differs from dicarlo.hvm)
    image_id = [x.split()[0][:-4] for x in open(data_dir.parent / 'image-metadata' / 'hvm_map.txt').readlines()]
    # Load neuroid related meta data
    neuroid_meta = pd.DataFrame(json.load(open(data_dir.parent / 'array-metadata' / 'mapping.json')))

    assembly = xr.DataArray(rate,
                            coords={'repetition': ('repetition', list(range(rate.shape[2]))),
                                    'time_bin_id': ('time_bin', list(range(rate.shape[0]))),
                                    'time_bin_start': ('time_bin', [x[0] for x in timebins]),
                                    'time_bin_stop': ('time_bin', [x[1] for x in timebins]),
                                    'image_id': ('image', image_id)},
                            dims=['time_bin', 'image', 'repetition', 'neuroid'])

    for column_name, column_data in neuroid_meta.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    assembly = assembly.sortby(assembly.image_id)
    stimuli = stimuli.sort_values(by='image_id').reset_index(drop=True)
    for column_name, column_data in stimuli.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('image', list(column_data.values))})
    assembly = assembly.sortby(assembly.id)  # Re-order by id to match dicarlo.hvm ordering

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = NeuronRecordingAssembly(assembly)

    # Filter noisy electrodes
    psth = np.load(data_dir / 'solo.rsvp.hvm.normalizer_psth.npy')
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

    # Add other experiment related info
    assembly.attrs['image_size_degree'] = 8
    assembly.attrs['stim_on_time_ms'] = 100

    return assembly


def main():
    data_dir = Path(__file__).parents[6] / 'data2' / 'active' / 'users' / 'sachis' / 'database'
    assert os.path.isdir(data_dir)

    stimuli = brainio_collection.get_stimulus_set('dicarlo.hvm')
    assembly = load_responses(data_dir, stimuli)
    assembly.name = 'dicarlo.Sanghavi2020'

    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio-brainscore')
    return


if __name__ == '__main__':
    main()
