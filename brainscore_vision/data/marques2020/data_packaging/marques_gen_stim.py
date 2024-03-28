import os
import numpy as np
from marques_stim_common import gen_grating_stim, gen_blank_stim, gen_texture_stim, \
    load_stim_info
from brainio_collection.packaging import package_stimulus_set

BLANK_STIM_NAME = 'Marques2020_blank'
RF_STIM_NAME = 'Marques2020_receptive_field'
ORIENTATION_STIM_NAME = 'Marques2020_orientation'
SF_STIM_NAME = 'Marques2020_spatial_frequency'
SIZE_STIM_NAME = 'Marques2020_size'
TEXTURE_STIM_NAME = 'FreemanZiemba2013_properties'

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_stimuli'
DEGREES = 12
SIZE_PX = 672
TEXTURE_DEGREES = 8

## All parameters
POS = np.array([0.5])
RADIUS = np.logspace(-3 + 0.75, 4 - 0.75, 12, endpoint=True, base=2) / 2
SF = np.logspace(-1.5 + 0.125, 4 - 0.125, 22, endpoint=True, base=2)
ORIENTATION = np.linspace(0, 165, 12, endpoint=True)
PHASE = np.linspace(0, 315, 8, endpoint=True)
CONTRAST = [1]

## Stimulus specific
POS_rf = np.linspace(-2.5, 2.5, 21, endpoint=True)
RADIUS_rf = [1/6]
SF_rf = [3]
ORIENTATION_rf = ORIENTATION[[0, 3, 6, 9]]
PHASE_rf = PHASE[[0, 4]]
RF_PARAMS = np.array(np.meshgrid(POS_rf, POS_rf, CONTRAST, PHASE_rf, ORIENTATION_rf, SF_rf,
                                 RADIUS_rf)).T.reshape(-1, 7)[:,[0,1,2,6,5,4,3]]

RADIUS_sf = np.array([0.75, 2.25])
ORIENTATION_sf = ORIENTATION[[0, 2, 4, 6, 8, 10]]
SF_PARAMS = np.array(np.meshgrid(POS, POS, CONTRAST, PHASE, ORIENTATION_sf, SF,
                                 RADIUS_sf)).T.reshape(-1, 7)[:,[0,1,2,6,5,4,3]]

SF_size = SF[[4, 8, 12, 16]]
ORIENTATION_size = ORIENTATION[[0, 2, 4, 6, 8, 10]]
SIZE_PARAMS = np.array(np.meshgrid(POS, POS, CONTRAST, PHASE, ORIENTATION_size, SF_size,
                                   RADIUS)).T.reshape(-1, 7)[:,[0,1,2,6,5,4,3]]

SF_or = SF[[4,8,12,16]]
MULT_RADIUS_or = [2, 3, 4]
SF_RADIUS_or = np.zeros((len(SF_or)*len(MULT_RADIUS_or), 2))
ind = 0
for mr in MULT_RADIUS_or:
    for sf in SF_or:
        SF_RADIUS_or[ind] = [mr/2/sf, sf]
        ind += 1
ORIENTATION_PHASE_or = np.array(np.meshgrid(ORIENTATION, PHASE)).T.reshape(-1, 2)
ORIENTATION_PARAMS = np.zeros((len(SF_RADIUS_or) * len(ORIENTATION_PHASE_or), 7))
ind = 0
for sf_rad in SF_RADIUS_or:
    for or_phase in ORIENTATION_PHASE_or:
        ORIENTATION_PARAMS[ind] = np.concatenate((POS, POS, CONTRAST, sf_rad, or_phase))
        ind += 1

STIM_NAMES = [RF_STIM_NAME, ORIENTATION_STIM_NAME, SF_STIM_NAME, SIZE_STIM_NAME]

GRAT_PARAMS = {RF_STIM_NAME: RF_PARAMS, ORIENTATION_STIM_NAME: ORIENTATION_PARAMS, SF_STIM_NAME: SF_PARAMS,
               SIZE_STIM_NAME: SIZE_PARAMS}


def main():
    blank_dir = DATA_DIR + os.sep + BLANK_STIM_NAME
    if not (os.path.isdir(blank_dir)):
        gen_blank_stim(degrees=DEGREES, size_px=448, save_dir=blank_dir)
    stimuli = load_stim_info(BLANK_STIM_NAME, blank_dir)
    print('Packaging stimuli:' + stimuli.identifier)
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')

    for stim_name in STIM_NAMES:
        stim_dir = DATA_DIR + os.sep + stim_name
        if not (os.path.isdir(stim_dir)):
            gen_grating_stim(degrees=DEGREES, size_px=SIZE_PX, stim_name=stim_name, grat_params=GRAT_PARAMS[stim_name],
                             save_dir=stim_dir)
        stimuli = load_stim_info(stim_name, stim_dir)
        print('Packaging stimuli:' + stimuli.identifier)
        package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')

    texture_dir = DATA_DIR + os.sep + TEXTURE_STIM_NAME
    if not (os.path.isdir(texture_dir)):
        gen_texture_stim(degrees=TEXTURE_DEGREES, stim_pos=np.array([POS[0], POS[0]]), save_dir=texture_dir)
    stimuli = load_stim_info(TEXTURE_STIM_NAME, texture_dir)
    print('Packaging stimuli:' + stimuli.identifier)
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.contrib')
    return


if __name__ == '__main__':
    main()
