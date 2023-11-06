from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm 

from brainio.assemblies import NeuroidAssembly
from brainio.packaging import package_data_assembly

SUBJECTS = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8"]

def collect_nsd_data_assembly(root_directory, subject):
    """
    Experiment information:
        - The 2023 Challenge data comes from the Natural Scenes Dataset (NSD) (Allen et al., 2022), 
          a massive 8-subjects dataset of 7T fMRI responses to images of natural scenes coming from 
          the COCO database (Lin et al., 2014).   

        - During the NSD experiment each subject viewed 10,000 distinct images, and a special set of 
          1000 images was shared across subjects (8 subjects x 9000 unique images + 1000 shared 
          images = 73,000 images). Each of the 10,000 images was presented three times, for a total of 30,000 
          image trials per subject. Subjects were instructed to focus on a fixation cross at the center of 
          the screen and performed a continuous recognition task in which they reported whether the current 
          image had been presented at any previous point in the experiment. All images were presented with 
          a visual angle of 8.4° x 8.4°.
    """

    # construct the assembly 
    subj = int(subject[-1])
    subject_data_dir = Path(f'{root_directory}/subj-data/subj0{subj}')
    train_dir = subject_data_dir / 'training_split'
    test_dir = subject_data_dir / 'test_split'
    roi_dir = subject_data_dir / 'roi_masks'

    train_fmri_dir = train_dir / 'training_fmri'
    test_fmri_dir = test_dir / 'test_fmri'
    ceiling_dir = test_dir / 'noise_ceiling'

    train_img_lookup = sorted(pd.read_csv(train_dir / 'imgs-lookup-train.csv', 
                                header = None, names = ['file_lookup'])['file_lookup'].tolist())
    test_img_lookup = sorted(pd.read_csv(test_dir / 'imgs-lookup-test.csv', 
                                header = None, names = ['file_lookup'])['file_lookup'].tolist())


    train_repetition = np.load(train_fmri_dir / 'repetition_train.npy')
    test_repetition = np.load(test_fmri_dir / 'repetition_test.npy')

    train_id = np.load(train_fmri_dir / 'train_id.npy')
    test_id = np.load(test_fmri_dir / 'test_id.npy')

    train_lh_fmri = np.load(train_fmri_dir / 'lh_training_fmri.npy')
    train_rh_fmri = np.load(train_fmri_dir / 'rh_training_fmri.npy')
    test_lh_fmri = np.load(test_fmri_dir / 'lh_test_fmri.npy')
    test_rh_fmri = np.load(test_fmri_dir / 'rh_test_fmri.npy')

    lh_fmri = np.concatenate((train_lh_fmri, test_lh_fmri), axis=0)
    rh_fmri = np.concatenate((train_rh_fmri, test_rh_fmri), axis=0)
    repetition = np.concatenate((train_repetition, test_repetition))
    img_cond = np.concatenate((['train']*len(train_id), ['test']*len(test_id)))
    stim_number = range(0, len(img_cond))

    counts_id = [0] * len(repetition)
    count = 0
    for i in range(1,len(repetition)):
        if repetition[i] - repetition[i-1] <= 0:
            count +=1
        counts_id[i] = count

    new_train_img_lookup = []
    assert(len(train_img_lookup) == len(np.unique(train_id)))
    for i, cond in enumerate(np.unique(train_id)):
        idx = np.where(train_id == cond)[0]
        for j in range(len(idx)):
            new_train_img_lookup.append(train_img_lookup[i])
    train_img_lookup = new_train_img_lookup
    del new_train_img_lookup

    new_test_img_lookup = []
    assert len(test_img_lookup) == len(np.unique(test_id))
    for i, cond in enumerate(np.unique(test_id)):
        idx = np.where(test_id == cond)[0]
        for j in range(len(idx)):
            new_test_img_lookup.append(test_img_lookup[i])
    test_img_lookup = new_test_img_lookup
    del new_test_img_lookup

    img_lookup = train_img_lookup + test_img_lookup
    assert(len(img_lookup) == len(train_img_lookup) + len(test_img_lookup))
    assert(len(img_lookup) == lh_fmri.shape[0])
    assert(len(img_lookup) == rh_fmri.shape[0])

    # load and concat the fmri
    fmri = np.concatenate((lh_fmri, rh_fmri), axis = 1)

    # extrapolate stimulus_id
    stimulus_id = [i.rsplit('_')[-1][:-4] for i in img_lookup]
    neuroid_id = np.arange(0, fmri.shape[1], 1) 
    hemisphere = np.concatenate((np.repeat('left', lh_fmri.shape[1]), np.repeat('right', rh_fmri.shape[1])), axis = 0)
    image_filename = [i.rsplit('/',1)[1] for i in img_lookup]

    # calculate the regions
    roi_classes = ['prf-visualrois','floc-faces', 'floc-bodies', 'floc-places', 'floc-words', 'streams']
    roi_data = {}

    for hem in ('left', 'right'):
        # 1. get the nsd_general_rsc_most_responsive
        specific_roi_dir = roi_dir / (hem[0]+'h.all-vertices_fsaverage_space.npy')
        fsaverage_all_vertices = np.load(specific_roi_dir)
        fsaverage_all_vertices_indices = np.where(fsaverage_all_vertices)[0]
        roi_map = {0: 'Unknown', 1: 'nsd_general_rsc_most_responsive'}
        result = [roi_map[value] for value in fsaverage_all_vertices]
        if hem == 'left':
            roi_data['fsaverage_all_vertices'] = [result[i] for i in fsaverage_all_vertices_indices]
        else:
            roi_data['fsaverage_all_vertices'] = roi_data['fsaverage_all_vertices'] +  [result[i] for i in fsaverage_all_vertices_indices]
        
        # 2. get the Noise ceiling
        lh_noise_ceiling = np.load(os.path.join(ceiling_dir, 'lh_noise_ceiling.npy'))
        rh_noise_ceiling = np.load(os.path.join(ceiling_dir, 'rh_noise_ceiling.npy')) 
        nc =   np.squeeze(np.concatenate((lh_noise_ceiling, rh_noise_ceiling)))
        del lh_noise_ceiling, rh_noise_ceiling

        # 3. get the rois 
        for roi_class in roi_classes:
            # Load the ROI brain surface maps
            roi_class_dir = roi_dir / (hem[0]+'h.'+roi_class+'_fsaverage_space.npy')
            roi_map_dir = roi_dir / ('mapping_'+roi_class+'.npy')
            fsaverage_roi_class = np.load(roi_class_dir)
            assert(np.all(np.isin(np.where(fsaverage_roi_class), fsaverage_all_vertices_indices)))
            roi_map = np.load(roi_map_dir, allow_pickle=True).item()
            
            result = [roi_map[value] for value in fsaverage_roi_class]
            if hem == 'left':
                roi_data[f'region_{roi_class}'] = [result[i] for i in fsaverage_all_vertices_indices]
            else:
                roi_data[f'region_{roi_class}'] = roi_data[f'region_{roi_class}'] + ([result[i] for i in fsaverage_all_vertices_indices])

    region_whole_brain=roi_data['fsaverage_all_vertices']
    region_prf_visualrois=roi_data['region_prf-visualrois']
    region_floc_faces=roi_data['region_floc-faces']
    region_floc_bodies=roi_data['region_floc-bodies']
    region_floc_places=roi_data['region_floc-places']
    region_floc_words=roi_data['region_floc-words']
    region_streams=roi_data['region_streams']
    

    coords = {
        'stimulus_id': ('presentation', stimulus_id), 
        'image_filename': ('presentation', image_filename),
        'image_folder': ('presentation', img_lookup),
        'repetition' : ('presentation', repetition),
        'algonauts_train_test' : ('presentation', img_cond),
        'stim_number' : ('presentation', stim_number),
        'counts_id' : ('presentation', counts_id),
        'neuroid_id': ('neuroid', neuroid_id),
        'hemisphere': ('neuroid', hemisphere),
        'region_whole_brain': ('neuroid', region_whole_brain),
        'region_prf_visualrois': ('neuroid', region_prf_visualrois),
        'region_floc_faces': ('neuroid', region_floc_faces),
        'region_floc_bodies': ('neuroid', region_floc_bodies),
        'region_floc_places': ('neuroid', region_floc_places),
        'region_floc_words': ('neuroid', region_floc_words),
        'region_streams': ('neuroid', region_streams),
        'algonauts_noise_ceiling' : ('neuroid', nc)
    }

    assembly = xr.DataArray(fmri,
                            coords=coords,
                            dims=['presentation', 'neuroid'])
   
    assembly = NeuroidAssembly(assembly)

    assembly = assembly.expand_dims('time_bin')
    assembly['time_bin_start'] = 'time_bin', [3000]
    assembly['time_bin_end'] = 'time_bin', [6000]
    assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')

    assembly.name = f'NSD2022_{subject}_assembly'

    return assembly


if __name__ == '__main__':

    root_directory = Path(r'./')

    for subject in tqdm(SUBJECTS, desc='Subjects'):

        assembly = collect_nsd_data_assembly(root_directory, subject)

        # upload to S3 
        stimulus_set_identifier = f'NSD2022_{subject}_stimulus_set'
        package_data_assembly('brainio_brainscore', assembly, assembly_identifier=assembly.name,
                            stimulus_set_identifier=stimulus_set_identifier,
                            assembly_class_name="NeuronRecordingAssembly", bucket_name="brainio-brainscore")