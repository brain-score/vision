# import libraries

# general
import numpy as np
import pandas as pd
import xarray as xr
import tqdm as tqdm


# brain-score specific
import brainscore
import brainio 
from brainscore.benchmarks._neural_common import average_repetition
from brainio.assemblies import NeuroidAssembly
from brainio.packaging import write_netcdf # use this function to save it locally
from brainio.packaging import package_data_assembly # use this function to push to S3
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

imgs_dir_path = '../images'
dependencies_dir_path = '../dependencies'

# oleo is already on S3
assembly_oleo = brainscore.get_assembly('dicarlo.Sanghavi2021.domain_transfer')

# pico is not on S3 yet, we load it from local 
file_path = dependencies_dir_path + '/data_pico/assy_dicarlo_pico_domain_transfer.nc'
assembly_pico = brainio.assemblies.DataAssembly.from_files(file_path)

# stimulus set with background id
csv_path = 'merged_stimulus_set.csv'
merged_stimulus_set = pd.read_csv(csv_path)

# need to add background id to the presentation dimension of oleo assembly
assembly_oleo['background_id'] = ('presentation', 
                                  np.array([merged_stimulus_set.background_id[merged_stimulus_set.stimulus_id == stim_id].values[0] 
                                        for stim_id in assembly_oleo['stimulus_id'].values]))

# small renaming and reordering of presentation and neuroid for assembly pico, to make it compatible with oleo
# we also add the backgroud_id to the presentation
data = assembly_pico.values
coords = {
    # selection of presentaiton
    'object_label': ('presentation', assembly_pico['object_label'].values),
    'object_style': ('presentation', assembly_pico['object_style'].values),
    'filepath': ('presentation', assembly_pico['filepath'].values),
    'stimulus_source': ('presentation', assembly_pico['identifier'].values),
    'image_file_name': ('presentation', assembly_pico['image_file_name'].values),
    'image_current_local_file_path': ('presentation', assembly_pico['image_current_local_file_path'].values),
    'image_id': ('presentation', assembly_pico['stimulus_id'].values),
    'repetition': ('presentation', assembly_pico['repetition'].values),
    'stimulus_id': ('presentation', assembly_pico['stimulus_id'].values),
    'filename': ('presentation', [assembly_pico['stimulus_id'].values[i] + '.png' for i in range(len(assembly_pico['stimulus_id'].values))]),
    'background_id' : ('presentation',
                    np.array([merged_stimulus_set.background_id[merged_stimulus_set.stimulus_id == stim_id].values[0] 
                              for stim_id in assembly_pico['stimulus_id'].values])),
    # selection of neuroid
    'col' : ('neuroid', assembly_pico['col'].values),
    'row' : ('neuroid', assembly_pico['row'].values),
    'bank' : ('neuroid', assembly_pico['bank'].values),
    'elec' : ('neuroid', assembly_pico['elec'].values),
    'label' : ('neuroid', assembly_pico['label'].values),
    'arr' : ('neuroid', assembly_pico['arr'].values),
    'hemisphere' : ('neuroid', assembly_pico['hemisphere'].values),
    'subregion' : ('neuroid', assembly_pico['subregion'].values),
    'region' : ('neuroid', assembly_pico['region'].values),
    'animal' : ('neuroid', assembly_pico['animal'].values),
    'neuroid_id' : ('neuroid', assembly_pico['neuroid_id'].values),
    # entire time_bin
    'time_bin' : assembly_pico.time_bin
}
assembly_pico = xr.DataArray(data, dims=['presentation', 'neuroid', 'time_bin'],
                        coords = coords)

assembly_pico = NeuroidAssembly(assembly_pico)

# some other operations for compatibility: we pad nans in Pico presentation coordinate to make sure that the number of repetitions is the same.:
new_data = np.empty((197694, 75, 7))
new_data[:] = np.nan

new_assembly_pico = xr.DataArray(new_data, dims=['presentation', 'neuroid', 'time_bin'],
                        coords = {
                            'presentation': assembly_oleo.presentation,
                            'neuroid' : assembly_pico.neuroid,
                            'time_bin' : assembly_pico.time_bin
                            })

for i in tqdm.tqdm(range(len(assembly_pico)), desc='merging'): # 

    r = i//35*28 +i
    new_assembly_pico[r] = assembly_pico[i]


# now we can finally merge the data:
merged_data = np.concatenate((assembly_oleo.values, new_assembly_pico.values), axis=1)

coords = {
    # entire presentation
    'presentation' : new_assembly_pico.presentation,
    # selection of neuroid
    'col' : ('neuroid', list(assembly_oleo['col'].values) + list(assembly_pico['col'].values)),
    'row' : ('neuroid', list(assembly_oleo['row'].values) + list(assembly_pico['row'].values)),
    'bank' : ('neuroid', list(assembly_oleo['bank'].values) + list(assembly_pico['bank'].values)),
    'elec' : ('neuroid', list(assembly_oleo['elec'].values)+ list(assembly_pico['elec'].values)),
    'label' : ('neuroid', list(assembly_oleo['label'].values) + list(assembly_pico['label'].values)),
    'arr' : ('neuroid', list(assembly_oleo['arr'].values) + list(assembly_pico['arr'].values)),
    'hemisphere' : ('neuroid', list(assembly_oleo['hemisphere'].values) + list(assembly_pico['hemisphere'].values)),
    'subregion' : ('neuroid', list(assembly_oleo['subregion'].values) + list(assembly_pico['subregion'].values)),
    'region' : ('neuroid', list(assembly_oleo['region'].values) + list(assembly_pico['region'].values)),
    'animal' : ('neuroid', list(assembly_oleo['animal'].values) + list(assembly_pico['animal'].values)),
    'electrode_id' : ('neuroid', list(assembly_oleo['neuroid_id'].values )+ list(assembly_pico['neuroid_id'].values)),
    'neuroid_id' : ('neuroid', list(assembly_oleo['animal'].values + '_' + assembly_oleo['neuroid_id'].values) + list(assembly_pico['animal'].values + '_' + assembly_pico['neuroid_id'].values)),
    
    # entire time_bin
    'time_bin' : assembly_oleo.time_bin
}

merged_assembly = xr.DataArray(merged_data,dims=['presentation', 'neuroid', 'time_bin'],
                            coords=coords)

merged_assembly = NeuroidAssembly(merged_assembly)
merged_assembly = merged_assembly.transpose('presentation', 'neuroid', 'time_bin')

merged_assembly.attrs = assembly_oleo.attrs

merged_assembly.attrs['identifier']='Igustibagus2024'
merged_assembly.attrs['stimulus_set']=merged_stimulus_set

merged_assembly.name = 'Igustibagus2024'

# package stimuli
stimuli = StimulusSet(merged_stimulus_set)
stimuli.stimulus_paths = {row['stimulus_id']: imgs_dir_path + '/' +  row['filename'] for _, row in stimuli.iterrows()}

stimuli.drop('filename', axis=1, inplace=True)

stimuli.name = 'Igustibagus2024'  # give the StimulusSet an identifier name

assert len(stimuli) == 3138  # make sure the StimulusSet is what you would expect

import pdb; pdb.set_trace()

package_stimulus_set(catalog_name='brainio_brainscore', proto_stimulus_set=stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore") # upload to S3

import pdb; pdb.set_trace()


# upload assmebly to S3                      
package_data_assembly('brainio_brainscore', merged_assembly, assembly_identifier=merged_assembly.name,
                      stimulus_set_identifier=stimuli.name,assembly_class_name="NeuronRecordingAssembly", 
                      bucket_name="brainio-brainscore")