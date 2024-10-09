import re, json
import numpy as np
import pandas as pd
import xarray as xr

from brainio.assemblies import NeuronRecordingAssembly
from brainio.stimuli import StimulusSet
from brainscore_vision import load_ceiling
from brainscore_vision.metric_helpers.transformations import CrossValidation
from pynwb.file import NWBFile
from typing import Tuple


def get_neuroids(nwb_file: NWBFile, subject: str) -> pd.DataFrame:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Get electrode metadata from nwb file.
    #-----------------------------------------------------------------------------------------------------------------------------
    data_list = []  
    for i in range(len(nwb_file.electrodes['location'])):
        data_dict       = {}
        location_item   = nwb_file.electrodes['location'][i]
        group_item      = nwb_file.electrodes['group'][i] 
        bank_item       = nwb_file.electrodes['group_name'][i] 
        label_item      = nwb_file.electrodes['label'][i]
        try: label_item = int(nwb_file.electrodes['label'][i].split('_')[0])
        except: label_item = nwb_file.electrodes['label'][i]

        location_match = re.search(r'\[(\d+).0, (\d+).0, (\d+).0\]', location_item)
        if location_match:
            data_dict['col'] = location_match.group(2)
            data_dict['row'] = location_match.group(1)
            data_dict['elec'] = location_match.group(3) 

        serialnumer = group_item.description.split('Serialnumber: ')[-1]
        data_dict['arr'] = serialnumer

        group_match = re.search(r"\['(\w+)', '(\w+)', '(\w+)'\]", group_item.location)
        if group_match:
            data_dict['hemisphere']  = group_match.group(1)
            data_dict['region']  = group_match.group(2)
            data_dict['subregion'] = group_match.group(3)
        
        data_dict['bank']  = bank_item.split('_')[-1]
        data_dict['animal'] = subject
        if (label_item) < 10:
            neuroid_id = f"{bank_item.split('_')[-1]}-00{label_item}"
            elec = f"00{label_item}"
        else:
            neuroid_id = f"{bank_item.split('_')[-1]}-0{label_item}"
            elec = f"00{label_item}"
        data_dict['neuroid_id']  = neuroid_id
        data_dict['elec']  = elec

        data_list.append(data_dict)
    
    neuroid_meta = pd.DataFrame(data_list)
    return neuroid_meta

def get_QC_neurids(nwb_file: NWBFile) -> Tuple[np.array, np.array]:
    '''
    This Method uses logical OR to find the common QC channels. (Closer to the BrainScore Method)
    '''
    psth = nwb_file.scratch['PSTHs_ZScored_SessionMerged'][:]
    common_QC_channels = np.logical_and.reduce(nwb_file.scratch['QualityApprovedChannelMasks'])
    channel_masks_day = nwb_file.scratch['QualityApprovedChannelMasks'][:]
    channel_mask_all_list = []
    day = 0
    for key in sorted(nwb_file.scratch.keys()):
        if key.startswith('PSTHs_QualityApproved_20'):
            nreps_per_day = nwb_file.scratch[key][:].shape[1]
            for i in range(nreps_per_day):
                channel_mask_all_list.append(channel_masks_day[day,:])
            day += 1
    channel_mask_all = np.array(channel_mask_all_list)

    assert channel_mask_all.shape[0] == psth.shape[1]
    filtered_neurids = np.any(channel_mask_all, axis=0)
    print(type(filtered_neurids), type(common_QC_channels))
    return filtered_neurids, common_QC_channels

def filter_neuroids(assembly: NeuronRecordingAssembly, threshold: float) -> NeuronRecordingAssembly:
    ceiler = load_ceiling('internal_consistency')
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly

def load_responses(nwb_file: NWBFile, json_file_path: str, 
                   stimuli: StimulusSet, use_QC_data: bool = True, 
                   do_filter_neuroids: bool = False, 
                   use_brainscore_filter_neuroids_method: bool = False) -> NeuronRecordingAssembly:
    #-----------------------------------------------------------------------------------------------------------------------------
    # Get the PSTH and normalizer PSTH
    #-----------------------------------------------------------------------------------------------------------------------------
    normalizer_psth = nwb_file.scratch['PSTHs_Normalizers_SessionMerged'][:]
    if not use_QC_data:
        psth            = nwb_file.scratch['PSTHs_ZScored_SessionMerged'][:]
    elif use_QC_data:
        psth            = nwb_file.scratch['PSTHs_QualityApproved_ZScored_SessionMerged'][:]
    with open(json_file_path, 'r') as f:
        params = json.load(f)
    meta = np.array([params['start_time_ms'], params['stop_time_ms'], params['tb_ms']], dtype=int)
    subject = params['subject']
    qc_array_or, qc_array_and = get_QC_neurids(nwb_file)
    #-----------------------------------------------------------------------------------------------------------------------------
    # Compute firing rates.
    #-----------------------------------------------------------------------------------------------------------------------------
    timebase = np.arange(meta[0], meta[1], meta[2])
    timebins = np.asarray([[int(x), int(x)+int(meta[2])] for x in timebase])
    assert len(timebase) == psth.shape[2]
    rate = np.empty((len(timebins), psth.shape[0], psth.shape[1], psth.shape[3]))
    for idx, tb in enumerate(timebins):
        t_cols = np.where((timebase >= (tb[0])) & (timebase < (tb[1])))[0]
        rate[idx] = np.mean(psth[:, :, t_cols, :], axis=2)  # Shaped time bins x images x repetitions x channels

    #-----------------------------------------------------------------------------------------------------------------------------
    # Load neuroid metadata and image metadata
    #-----------------------------------------------------------------------------------------------------------------------------
    image_id     = stimuli.image_number
    neuroid_meta = get_neuroids(nwb_file, subject)

    assembly = xr.DataArray(rate,
                            coords={'repetition': ('repetition', list(range(rate.shape[2]))),
                                    'neuroid_id': ('neuroid', list(range(psth.shape[3]))),
                                    'region': ('neuroid', ['IT'] * psth.shape[3]),
                                    'time_bin_id': ('time_bin', list(range(rate.shape[0]))),
                                    'time_bin_start': ('time_bin', [x[0] for x in timebins]),
                                    'time_bin_stop': ('time_bin', [x[1] for x in timebins]),
                                    'image_id': ('image', image_id)},
                            dims=['time_bin', 'image', 'repetition', 'neuroid'])
    if use_QC_data:
        for column_name, column_data in neuroid_meta.items():
            assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values[qc_array_and]))})
    else:
        for column_name, column_data in neuroid_meta.items():
            assembly = assembly.assign_coords(**{f'{column_name}': ('neuroid', list(column_data.values))})

    assembly = assembly.sortby(assembly.image_id)
    assembly.name = stimuli.name
    stimuli  = stimuli.sort_values(by='image_id').reset_index(drop=True)
    for column_name, column_data in stimuli.items():
        assembly = assembly.assign_coords(**{f'{column_name}': ('image', list(column_data.values))})
    assembly = assembly.sortby(assembly.id)  

    # Collapse dimensions 'image' and 'repetitions' into a single 'presentation' dimension
    assembly = assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
    assembly = NeuronRecordingAssembly(assembly)

    if do_filter_neuroids and use_brainscore_filter_neuroids_method:
        # Filter noisy electrodes
        psth = normalizer_psth
        if psth.shape[0] == 26:
            psth = psth[:-1,:,:,:] #remove grey image
        t_cols = np.where((timebase >= (70 )) & (timebase < (170)))[0]
        rate = np.mean(psth[:, :, t_cols, :], axis=2)
        normalizer_assembly = xr.DataArray(rate,
                                        coords={'repetition': ('repetition', list(range(rate.shape[1]))),
                                                'image_id': ('image', list(range(rate.shape[0]))),
                                                'id': ('image', list(range(rate.shape[0])))},
                                        dims=['image', 'repetition', 'neuroid'])
        for column_name, column_data in neuroid_meta.items():
            normalizer_assembly = normalizer_assembly.assign_coords(
                **{f'{column_name}': ('neuroid', list(column_data.values))})

        normalizer_assembly = normalizer_assembly.assign_coords(**{f'{"stimulus_id"}': ('image', list(np.linspace(1,psth.shape[0],psth.shape[0], dtype=int)))})# had to add this part TODO: remove the last grey image from normalizer set when doing the nwb conversion
        normalizer_assembly = normalizer_assembly.stack(presentation=('image', 'repetition')).reset_index('presentation')
        normalizer_assembly = normalizer_assembly.drop('image')
        normalizer_assembly = normalizer_assembly.transpose('presentation', 'neuroid')
        normalizer_assembly = NeuronRecordingAssembly(normalizer_assembly)

        filtered_assembly = filter_neuroids(normalizer_assembly, 0.7)
        assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))

    elif do_filter_neuroids and not use_brainscore_filter_neuroids_method:
        filter_assembly = xr.DataArray(qc_array_or,
                                dims=['neuroid'])
        for column_name, column_data in neuroid_meta.items():
            filter_assembly = filter_assembly.assign_coords(
                **{f'{column_name}': ('neuroid', list(column_data.values))})
            
        filtered_assembly = filter_assembly.sel(neuroid=qc_array_or)
        assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))

    elif use_QC_data:
        filter_assembly = xr.DataArray(qc_array_and,
                                dims=['neuroid'])
        for column_name, column_data in neuroid_meta.items():
            filter_assembly = filter_assembly.assign_coords(
                **{f'{column_name}': ('neuroid', list(column_data.values))})
            
        filtered_assembly = filter_assembly.sel(neuroid=qc_array_and)
        assembly = assembly.sel(neuroid=np.isin(assembly.neuroid_id, filtered_assembly.neuroid_id))

    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    # Add other experiment related info
    assembly.attrs['image_size_degree'] = 8
    assembly.attrs['stim_on_time_ms'] = 100

    assembly.attrs['stimulus_set'] = stimuli

    return assembly
