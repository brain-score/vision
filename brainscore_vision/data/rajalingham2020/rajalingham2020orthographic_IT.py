import os
from pathlib import Path
import pickle

import numpy as np
import xarray as xr

from brainio_base.stimuli import StimulusSet
from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_collection.packaging import package_stimulus_set, package_data_assembly


def collect_stimuli(data_dir):
    IT_base616 = pickle.load(open(os.path.join(data_dir, 'data_IT_base616.pkl'), 'rb'))
    stimuli = IT_base616['meta']

    stimuli = stimuli.rename(columns={'id': 'image_id'})

    stimuli['image_file_name'] = ''
    stimuli['image_current_local_file_path'] = ''

    for idx, row in stimuli.iterrows():
        image_file_name = f'{row.image_id}.png'
        image_file_path = os.path.join(data_dir, 'stimuli', image_file_name)

        stimuli.at[idx, 'image_file_name'] = image_file_name
        stimuli.at[idx, 'image_current_local_file_path'] = image_file_path

    stimuli['grp5_bigram_freq'] = stimuli['grp5_bigram_freq'].astype(str)  # IntervalIndex not supported by netCDF4
    stimuli = stimuli.astype({column_name: 'int32' for column_name
                              in stimuli.select_dtypes(include=['bool']).keys()})  # Bool not supported by netCDF4
    assert len(np.unique(stimuli['image_id'])) == len(stimuli)
    stimuli = StimulusSet(stimuli)
    stimuli.image_paths = \
        {stimuli.at[idx, 'image_id']: stimuli.at[idx, 'image_current_local_file_path'] for idx in range(len(stimuli))}
    return stimuli


def load_responses(data_dir, stimuli):
    IT_base616 = pickle.load(open(os.path.join(data_dir, 'data_IT_base616.pkl'), 'rb'))
    features = IT_base616['IT_features']  # Shaped images x neuroids x repetitions x time_bins

    # Drop all time_bins except the fifth, which corresponds to 70-170ms
    # For future reference the time-bins are as follows:
    # 70-120ms, 120-170ms, 170-220ms, 220-270ms, 70-170ms, 170-270ms, 70-270ms
    features = features[:, :, :, 4]
    features = features[:, :, :, np.newaxis]
    # Drop all repetitions beyond 33rd (all neuroids have at least 27, but none have greater than 33)
    features = features[:, :, :33, :]

    neuroid_meta = pickle._Unpickler(open(os.path.join(data_dir, 'IT_neural_meta_full.pkl'), 'rb'))
    neuroid_meta.encoding = 'latin1'
    neuroid_meta = neuroid_meta.load()

    assembly = xr.DataArray(features,
                            coords={'region': ('neuroid', ['IT'] * len(neuroid_meta)),
                                    'neuroid_id': ('neuroid', list(range(features.shape[1]))),
                                    'time_bin_start': ('time_bin', [70]),
                                    'time_bin_stop': ('time_bin', [170]),
                                    'repetition': ('repetition', list(range(features.shape[2])))},
                            dims=['image', 'neuroid', 'repetition', 'time_bin'])

    for column_name, column_data in neuroid_meta.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    for column_name, column_data in stimuli.iteritems():
        assembly = assembly.assign_coords(**{f'{column_name}': ('image', list(column_data.values))})

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = assembly.drop('image')

    assembly = NeuronRecordingAssembly(assembly)
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    return assembly


def main():
    data_dir = Path(__file__).parents[5] / 'data2' / 'active' / 'users' / 'sachis' / 'database' / 'Rajalingham2020'
    assert os.path.isdir(data_dir)
    stimuli = collect_stimuli(data_dir)
    stimuli.identifier = 'Rajalingham2020'
    assembly = load_responses(data_dir, stimuli)
    assembly.name = 'dicarlo.Rajalingham2020'

    print('Packaging stimuli')
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')
    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio.dicarlo')

    return


if __name__ == '__main__':
    main()
