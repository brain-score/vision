import glob, os, re
import numpy as np
import xarray as xr

from brainio.assemblies import NeuronRecordingAssembly
from brainio.stimuli import StimulusSet
from brainio.packaging import package_data_assembly
from IPython.display import display
from pynwb import NWBHDF5IO

# hardcoded based on dandiset
DANDISET_NUMBER = '000812'
IMAGE_SET = 'IAPS'

def create_neural_assembly(psth, meta):
    timebase = np.arange(meta[0], meta[1], meta[2])
    timebins = np.asarray([[int(x), int(x)+int(meta[2])] for x in timebase])
    assert len(timebase) == psth.shape[2], f"Number of bins is not correct. Expected {len(timebase)} got {psth.shape[2]}"


    assembly = xr.DataArray(psth,
                    coords={'repetition': ('repetition', list(range(psth.shape[1]))),
                            'stimulus_id': ('image', list(range(psth.shape[0]))),
                            'time_bin_id': ('time_bin', list(range(psth.shape[2]))),
                            'neuroid_id': ('neuroid', list(range(psth.shape[3]))),
                            'region': ('neuroid', ['IT'] * psth.shape[3]),
                            'time_bin_start': ('time_bin', [x[0] for x in timebins]),
                            'time_bin_stop': ('time_bin', [x[1] for x in timebins])},
                    dims=['image', 'repetition', 'time_bin', 'neuroid'])

    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = NeuronRecordingAssembly(assembly)

    return assembly


# root_dir  = '/braintree/home/aliya277/dandi_folder_test'
root_dir = '/braintree/home/aliya277/dandi_folder_train'
test_train = 'Train'

# experiment_file_paths = glob.glob(os.path.join(root_dir, '*'))
# for experiment_path in sorted(experiment_file_paths)[0:1]:
#     print(os.path.basename(experiment_path))

experiment_path = f"/Users/caroljiang/Downloads/vision/brainscore_vision/data/data_generation_trial/{DANDISET_NUMBER}"

# ImageSet        = os.path.basename(experiment_path)
# nwb_file_name   = os.listdir(os.path.join(experiment_path, f"{ImageSet}.sub_pico"))[0]
# nwb_file_path   = os.path.join(os.path.join(experiment_path, f"{ImageSet}.sub_pico", nwb_file_name))
ImageSet        = IMAGE_SET
nwb_file_name   = os.listdir(os.path.join(experiment_path, "sub-pico"))[0]
nwb_file_path   = os.path.join(os.path.join(experiment_path, "sub-pico", nwb_file_name))

print("Loading the NWB file ...")
io = NWBHDF5IO(nwb_file_path, "r")
nwb_file = io.read()

def output_assembly():
    print(nwb_file.scratch.keys())
    if 'CombinedQualityCheckedPSTHs' in nwb_file.scratch.keys():
        psth = nwb_file.scratch['CombinedQualityCheckedPSTHs'][:]
        psth_meta = nwb_file.scratch['CombinedQualityCheckedPSTHs'].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1]
    # not part of aliya's code, added for 000781 (and 000812), ask
    elif 'PSTHs_QualityApproved_SessionMerged' in nwb_file.scratch.keys():
        psth = nwb_file.scratch['PSTHs_QualityApproved_SessionMerged'][:]
        psth_meta = nwb_file.scratch['PSTHs_QualityApproved_SessionMerged'].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1]
    else: # if there is only one PSTH, then there is no Combined PSTH
        for key in nwb_file.scratch.keys():
            print('k', key)
            if key.startswith('QualityCheckedPSTH_'):
                psth = nwb_file.scratch[key][:]
                psth_meta = nwb_file.scratch[key].description.split('[start_time_ms, stop_time_ms, tb_ms]: ')[-1]

    psth_meta = re.findall(r'\d+', psth_meta)
    psth_meta = [int(i) for i in psth_meta]

    io.close()

    assembly = create_neural_assembly(psth, psth_meta)
    assembly.name = f"{ImageSet}"

    display(assembly)

    return assembly


if __name__ == '__main__':
    a = output_assembly()