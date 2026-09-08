
import numpy as np
import brainscore_vision

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_stimulus_set_locally
from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_core.supported_data_standards.brainio import packaging
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

import statistics


# Note:
# Running local on mac

session_id = 190923 # 190923 201025 210225 211022
date_experiment = '2019-09-23' # '2019-09-23' '2020-10-25' '2021-02-25' '2021-10-22'

responses = np.load('/Users/cowley/Desktop/brainscore_upload/data_raw/responses_{:d}.npy'.format(session_id))
	# (num_neurons, num_images, num_possible_repeats)

num_neurons = responses.shape[0]
num_images = responses.shape[1]


## STIMULUS SET

# Create a dataframe tracking image paths and attributes
print('---STIMULUS SET---')
stimuli_data = [{'stimulus_id': 'image{:04d}'.format(iimage), 'object_name': '{:04d}'.format(iimage)} for iimage in range(1,num_images+1)]
stimulus_set = StimulusSet(stimuli_data)

stimulus_set.stimulus_paths = {
	'image{:04d}'.format(iimage): '/Users/cowley/Desktop/brainscore_upload/data_raw/images_{:d}/image{:04d}.jpg'.format(session_id, iimage)
	for iimage in range(1,num_images+1)
	}

stimulus_set.name = 'Cowley2026.{:d}'.format(session_id)

package_output = package_stimulus_set_locally(
    proto_stimulus_set=stimulus_set,
    stimulus_set_identifier=stimulus_set.name,
)

print(package_output)






## NEURAL DATA

print()
print('---NEURAL DATA---')


## flatten responses to be (num_neurons, num_repeats*num_images)

responses_data_matrix = []
stimulus_ids = []
object_names = []
repeat_ids = []


for iimage in range(num_images):
	num_repeats = np.sum(~np.isnan(responses[0,iimage,:]))

	for irepeat in range(num_repeats):

		responses_data_matrix.append(responses[:,iimage,irepeat])
		stimulus_ids.append('image{:04d}'.format(iimage+1))
		object_names.append('{:04d}'.format(iimage+1))
		repeat_ids.append(irepeat)


responses_data_matrix = np.stack(responses_data_matrix)
	# (num_presentations, num_neurons)
responses_data_matrix = np.expand_dims(responses_data_matrix, axis=2)  # include time_bin dimension (dummy)


assembly = NeuroidAssembly(
    responses_data_matrix,
    coords={
        # Coordinates tracking the 'presentation' dimension (axis 1)
        'stimulus_id': ('presentation', stimulus_ids),
        'object_name': ('presentation', object_names),
        'repetition': ('presentation', repeat_ids),

        # Coordinates tracking the 'neuroid' dimension (axis 0)
        'neuroid_id': ('neuroid', [f'neuron_{i}' for i in range(num_neurons)]),
        'region': ('neuroid', ['V4'] * num_neurons),  # e.g., 'V4', 'IT', or 'AL'

        'time_bin_start': ('time_bin', [50]),
        'time_bin_end': ('time_bin', [150])
    },
    dims=['presentation', 'neuroid', 'time_bin']
)


assembly.attrs['experiment_date'] = date_experiment
assembly.name = 'Cowley2026.{:d}'.format(session_id)


package_output = packaging.package_data_assembly_locally(
	proto_data_assembly=assembly,
	assembly_identifier=stimulus_set.name, # We use the same stimulusSet name for the assembly
	stimulus_set_identifier=stimulus_set.name,
	assembly_class_name="NeuroidAssembly", # For most neural data, use NeuroidAssembly. For behavioral data, use BehavioralAssembly
)

print(package_output)


