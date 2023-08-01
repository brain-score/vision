import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import re
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from brainio_base.stimuli import StimulusSet
from brainio_base.assemblies import NeuronRecordingAssembly


from brainio_collection.packaging import package_stimulus_set, package_data_assembly

storage_location = ("C:/Users/hsuen/Desktop/bigData/brainscore_img_elec_time_70hz150/")


# This needs to return a stimulus set
# Which will be one row per image
# Three columns, image ID and image filename and (additional) the label

def collect_stimuli(stimuli_directory):
    labels = np.load(stimuli_directory + 'stimgroups.npy')  # labels of image
    stim_sequence = np.load(
        stimuli_directory + 'stimsequence.npy')  # the names of the files with a b: "b'V12'" (Image ID)
    # image file name will be the stimuli_directory + ID.jpg

    stimuli = []
    for x in range(len(labels)):
        stimuli.append({
            'image_id': stim_sequence[x].decode('UTF-8'),  # extract just the ID
            'image_file_name': stimuli_directory + "stimuli/" + str(stim_sequence[x].decode('UTF-8')) + ".jpg",
            'image_number': x,
            'label': labels[x],
        })
    stimuli = pd.DataFrame(stimuli)



    # convert stimuli object into something that can be used with all the packaging functions
    stimuli = StimulusSet(stimuli)

    # after converted to a type "StimulusSet", you set an attribute of the object, suchas "image_paths":
    stimuli.image_paths = {key: stimuli['image_file_name'][i] for i, key in enumerate(stimuli['image_id'])}

    return stimuli


# pass into this function the stimuli object that you obtain from the above function
# stimuli is a Pandas DataFrame
# also pass into this function the neural response file (neural_responses.npy)
def load_responses(response_file, stimuli):
    neural_response_file = response_file + "neural_responses.npy"
    neural_responses = np.load(neural_response_file)

    brodmann_file = response_file + "brodmann_areas.npy"
    brodmann_locations = np.load(brodmann_file)

    assembly = xr.DataArray(neural_responses,
                            coords={
                                'image_num': ('presentation', list(range(neural_responses.shape[0]))),
                                'image_id': ('presentation',
                                             [stimuli['image_id'][stimuli['image_number'] == num].values[0]
                                              for num in range(neural_responses.shape[0])]),

                                'region': ('neuroid', brodmann_locations),
                                # right now puts value "brodmann" area for all coords

                                'neuroid_id': ('neuroid', list(range(neural_responses.shape[1]))),

                                'time': ('time_bin', np.linspace(0, 1, 32)),
                                'time_bin_start': ('time_bin', np.arange(0, 1000, 31.25)),
                                'time_bin_end': ('time_bin', np.arange(31.25, 1001, 31.25))
                            },
                            dims=['presentation', 'neuroid', 'time_bin'])

    assembly = NeuronRecordingAssembly(assembly)


    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
    return assembly


def main():
    stimuli = collect_stimuli(storage_location)
    stimuli.name = 'aru.Kuzovkin2018'

    assembly = load_responses(storage_location, stimuli)
    assembly.name = 'aru.Kuzovkin2018'

    print("Packaging stimuli")
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.name,
                         bucket_name="brainio.contrib")
    print("Packaging assembly")
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=stimuli.name,
                          bucket_name="brainio.contrib")


if __name__ == '__main__':
    main()
