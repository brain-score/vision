
import numpy as np
import imageio
import os
import pandas as pd
from glob import glob

import matplotlib.pyplot as plt
from brainio_base.stimuli import StimulusSet


class Stimulus:
    def __init__(self, size_px=[448, 448], bit_depth=8,
                 stim_id=1000, save_dir='images', type_name='stimulus',
                 format_id='{0:04d}'):
        self.save_dir = save_dir
        self.stim_id = stim_id
        self.format_id = format_id
        self.type_name = type_name

        self.white = np.uint8(2**bit_depth-1)
        self.black = np.uint8(0)
        self.gray = np.uint8(self.white/2+1)
        self.size_px = size_px
        self.objects = []
        self.stimulus = np.ones(self.size_px, dtype=np.uint8) * self.gray

    def add_object(self, stim_object):
        self.objects.append(stim_object)

    def build_stimulus(self):
        for obj in self.objects:
            self.stimulus[obj.mask] = obj.stimulus[obj.mask]

    def clear_stimulus(self):
        self.stimulus = np.ones(self.size, dtype=np.uint8) * self.gray

    def show_stimulus(self):
        my_dpi = 192
        fig = plt.figure()
        fig.set_size_inches(self.size_px[1] / my_dpi, self.size_px[0] / my_dpi, forward=False)
        ax = plt.axes([0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.stimulus, cmap='gray')
        plt.show()

    def save_stimulus(self):
        file_name= self.type_name + '_' + self.format_id.format(self.stim_id) + '.png'
        imageio.imwrite(self.save_dir + os.sep + file_name, self.stimulus)
        return file_name


class Grating:
    def __init__(self, orientation=0, phase=0, sf=2, size_px=[448, 448], width=8,
                 contrast=1, bit_depth=8, pos=[0, 0], rad=5, sig=0,
                 stim_id=1000, format_id='{0:04d}', save_dir='images', type_name='grating'):

        # save directory
        self.save_dir = save_dir
        self.stim_id = stim_id
        self.format_id = format_id
        # label for type of stimulus
        self.type_name = type_name

        # 1 channel colors, white, black, grey
        self.white = np.uint8(2**bit_depth-1)
        self.black = np.uint8(0)
        self.gray = np.uint8(self.white/2+1)

        # pixel dimensions of the image
        self.size_px = np.array(size_px)
        # position of image in field of view
        self.pos = np.array(pos)
        # pixel to visual field degree conversion
        self.px_to_deg = self.size_px[1] / width
        # size of stimulus in visual field in degrees
        self.size = self.size_px / self.px_to_deg

        # orientation in radians
        self.orientation = orientation / 180 * np.pi
        # phase of the grating
        self.phase = phase / 180 * np.pi
        # spatial frequency of the grating
        self.sf = sf
        # contrast of the grating
        self.contrast = contrast

        # make self.xv and self.yv store the degree positions of all pixels in the image
        self.xv = np.zeros(size_px)
        self.yv = np.zeros(size_px)
        self.update_frame()

        self.mask = np.ones(size_px, dtype=bool)
        self.set_circ_mask(rad=rad)

        self.tex = np.zeros(size_px)
        self.stimulus = np.ones(size_px, dtype=np.uint8) * self.gray

        self.envelope = np.ones(size_px)
        if sig is 0:
            self.update_tex()
        else:
            self.set_gaussian_envelope(sig)

    def update_frame(self):
        x = (np.arange(self.size_px[1]) - self.size_px[1]/2) / self.px_to_deg - self.pos[1]
        y = (np.arange(self.size_px[0]) - self.size_px[0]/2) / self.px_to_deg - self.pos[0]

        # all possible degree coordinates in matrices of points
        self.xv, self.yv = np.meshgrid(x, y)

    def update_tex(self):
        # make the grating pattern
        self.tex = (np.sin((self.xv * np.cos(self.orientation) + self.yv * np.sin(self.orientation)) *
                           self.sf * 2 * np.pi + self.phase) * self.contrast * self.envelope)

    def update_stimulus(self):
        self.stimulus[self.mask] = np.uint8(((self.tex[self.mask]+1)/2)*self.white)
        self.stimulus[np.logical_not(self.mask)] = self.gray

    def set_circ_mask(self, rad):
        # apply operation to put a 1 for all points inclusively within the degree radius and a 0 outside it
        self.mask = self.xv**2 + self.yv**2 <= rad ** 2

    # same as circular mask but for an annulus
    def set_annular_mask(self, inner_rad, outer_rad):
        self.mask = (self.xv ** 2 + self.yv ** 2 <= outer_rad ** 2) * \
                    (self.xv ** 2 + self.yv ** 2 > inner_rad ** 2)

    def set_gaussian_envelope(self, sig):
        d = np.sqrt(self.xv**2 + self.yv**2)
        self.envelope = np.exp(-d**2/(2 * sig**2))
        self.update_tex()

    def show_stimulus(self):
        # pyplot stuff
        self.update_stimulus()
        my_dpi = 192
        fig = plt.figure()
        fig.set_size_inches(self.size_px[1] / my_dpi, self.size_px[0] / my_dpi, forward=False)
        ax = plt.axes([0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.stimulus, cmap='gray')
        plt.show()

    def save_stimulus(self):
        # save to correct (previously specified) directory
        self.update_stimulus()
        file_name = self.type_name + '_' + self.format_id.format(self.stim_id) + '.png'
        imageio.imwrite(self.save_dir + os.sep + file_name, self.stimulus)
        return file_name


def load_stim_info(stim_name, data_dir):
    stim = pd.read_csv(os.path.join(data_dir, 'stimulus_set'), dtype={'image_id': str})
    image_paths = dict((key, value) for (key, value) in zip(stim['image_id'].values,
                                                            [os.path.join(data_dir, image_name) for image_name
                                                             in stim['image_file_name'].values]))
    stim_set = StimulusSet(stim[stim.columns[:-1]])
    stim_set.image_paths = image_paths
    stim_set.identifier = stim_name

    return stim_set


def gen_blank_stim(degrees, size_px, save_dir):
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    stim = Stimulus(size_px=[size_px, size_px], type_name='blank_stim', save_dir=save_dir, stim_id=0)

    stimuli = pd.DataFrame({'image_id': str(0), 'degrees': [degrees]})
    image_names = (stim.save_stimulus())

    stimuli['image_file_name'] = pd.Series(image_names)
    stimuli['image_current_local_file_path'] = pd.Series(save_dir + os.sep + image_names)

    stimuli.to_csv(save_dir + os.sep + 'stimulus_set', index=False)


def gen_grating_stim(degrees, size_px, stim_name, grat_params, save_dir):
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    width = degrees

    nStim = grat_params.shape[0]

    print('Generating stimulus: #', nStim)

    stimuli = pd.DataFrame({'image_id': [str(n) for n in range(nStim)], 'degrees': [width] * nStim})

    image_names = nStim * [None]
    image_local_file_path = nStim * [None]
    all_y = nStim * [None]
    all_x = nStim * [None]
    all_c = nStim * [None]
    all_r = nStim * [None]
    all_s = nStim * [None]
    all_o = nStim * [None]
    all_p = nStim * [None]

    for i in np.arange(nStim):
        stim_id = np.uint64(grat_params[i, 0] * 10e9 + grat_params[i, 1] * 10e7 + grat_params[i, 3] * 10e5 +
                            grat_params[i, 4] * 10e3 + grat_params[i, 5] * 10e1 + grat_params[i, 6])
        grat = Grating(width=width, pos=[grat_params[i, 0], grat_params[i, 1]], contrast=grat_params[i, 2],
                       rad=grat_params[i, 3], sf=grat_params[i, 4], orientation=grat_params[i, 5],
                       phase=grat_params[i, 6], stim_id= stim_id, format_id='{0:012d}', save_dir=save_dir,
                       size_px=[size_px, size_px], type_name=stim_name)
        image_names[i] = (grat.save_stimulus())
        image_local_file_path[i] = save_dir + os.sep + image_names[i]
        all_y[i] = grat_params[i, 0]
        all_x[i] = grat_params[i, 1]
        all_c[i] = grat_params[i, 2]
        all_r[i] = grat_params[i, 3]
        all_s[i] = grat_params[i, 4]
        all_o[i] = grat_params[i, 5]
        all_p[i] = grat_params[i, 6]

    stimuli['position_y'] = pd.Series(all_y)
    stimuli['position_x'] = pd.Series(all_x)
    stimuli['contrast'] = pd.Series(all_c)
    stimuli['radius'] = pd.Series(all_r)
    stimuli['spatial_frequency'] = pd.Series(all_s)
    stimuli['orientation'] = pd.Series(all_o)
    stimuli['phase'] = pd.Series(all_p)
    stimuli['image_file_name'] = pd.Series(image_names)
    stimuli['image_current_local_file_path'] = pd.Series(image_local_file_path)

    stimuli.to_csv(save_dir + os.sep + 'stimulus_set', index=False)


def gen_grating_stim_old(degrees, size_px, stim_name, grat_contrast, grat_pos, grat_rad, grat_sf, grat_orientation,
                     grat_phase, save_dir):
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    width = degrees

    nStim = len(grat_pos) * len(grat_pos) * len(grat_contrast) * len(grat_rad) * len(grat_sf) * len(grat_orientation) \
            * len(grat_phase)

    print('Generating stimulus: #', nStim)

    stimuli = pd.DataFrame({'image_id': [str(n) for n in range(nStim)], 'degrees': [width] * nStim})

    image_names = nStim * [None]
    image_local_file_path = nStim * [None]
    all_y = nStim * [None]
    all_x = nStim * [None]
    all_c = nStim * [None]
    all_r = nStim * [None]
    all_s = nStim * [None]
    all_o = nStim * [None]
    all_p = nStim * [None]

    i = 0
    for y in np.arange(len(grat_pos)):
        for x in np.arange(len(grat_pos)):
            for c in np.arange(len(grat_contrast)):
                for r in np.arange(len(grat_rad)):
                    for s in np.arange(len(grat_sf)):
                        for o in np.arange(len(grat_orientation)):
                            for p in np.arange(len(grat_phase)):
                                grat = Grating(width=width, pos=[grat_pos[y], grat_pos[x]],
                                               contrast=grat_contrast[c], rad=grat_rad[r],
                                               sf=grat_sf[s], orientation=grat_orientation[o],
                                               phase=grat_phase[p],
                                               stim_id=np.uint64(
                                                   y * 10e9 + x * 10e7 + r * 10e5 + s * 10e3 + o * 10e1 + p),
                                               format_id='{0:012d}', save_dir=save_dir, size_px=[size_px, size_px],
                                               type_name=stim_name)
                                image_names[i] = (grat.save_stimulus())
                                image_local_file_path[i] = save_dir + os.sep + image_names[i]
                                all_y[i] = grat_pos[y]
                                all_x[i] = grat_pos[x]
                                all_c[i] = grat_contrast[c]
                                all_r[i] = grat_rad[r]
                                all_s[i] = grat_sf[s]
                                all_o[i] = grat_orientation[o]
                                all_p[i] = grat_phase[p]
                                i += 1

    stimuli['position_y'] = pd.Series(all_y)
    stimuli['position_x'] = pd.Series(all_x)
    stimuli['contrast'] = pd.Series(all_c)
    stimuli['radius'] = pd.Series(all_r)
    stimuli['spatial_frequency'] = pd.Series(all_s)
    stimuli['orientation'] = pd.Series(all_o)
    stimuli['phase'] = pd.Series(all_p)
    stimuli['image_file_name'] = pd.Series(image_names)
    stimuli['image_current_local_file_path'] = pd.Series(image_local_file_path)

    stimuli.to_csv(save_dir + os.sep + 'stimulus_set', index=False)


def gen_texture_stim(degrees, stim_pos, save_dir):
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    original_dir = '/braintree/home/tmarques/.brainio/image_movshon_stimuli/movshon_stimuli'

    gray_c = 128

    input_degrees = 4
    aperture_degrees = 4

    original_filenames = []
    for image_file_path in glob(f"{original_dir}/*.png"):
        original_filenames.append(image_file_path)

    nStim = len(original_filenames)

    im_temp = imageio.imread(original_filenames[0])

    # Image size
    size_px = np.array(im_temp.shape).astype(int)
    px_deg = size_px[0] / input_degrees

    size_px_out = (size_px * (degrees / input_degrees)).astype(int)
    cnt_px = (stim_pos * px_deg).astype(int)

    size_px_disp = ((size_px_out - size_px) / 2).astype(int)

    fill_ind = [[(size_px_disp[0] + cnt_px[0]), (size_px_disp[0] + cnt_px[0] + size_px[0])],
                [(size_px_disp[1] + cnt_px[1]), (size_px_disp[1] + cnt_px[1] + size_px[1])]]

    # Image aperture
    a = aperture_degrees * px_deg / 2
    # Meshgrid with pixel coordinates
    x = (np.arange(size_px_out[1]) - size_px_out[1] / 2)
    y = (np.arange(size_px_out[0]) - size_px_out[0] / 2)
    xv, yv = np.meshgrid(x, y)
    # Raised cosine aperture
    inner_mask = (xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2 < a ** 2
    cos_mask = 1 / 2 * (1 + np.cos(np.sqrt((xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2) / a * np.pi))
    cos_mask[np.logical_not(inner_mask)] = 0

    familyNumOrder = np.array([60, 56, 13, 48, 71, 18, 327, 336, 402, 38, 23, 52, 99, 393, 30])

    type = []
    sample = []
    family = []
    position_x = []
    position_y = []
    image_names = nStim * [None]
    image_local_file_path = nStim * [None]

    for n in range(nStim):
        im = imageio.imread(original_filenames[n])
        im = im - gray_c * np.ones(size_px)
        im_template = np.zeros(size_px_out)
        im_template[fill_ind[0][0]:fill_ind[0][1], fill_ind[1][0]:fill_ind[1][1]] = im
        im_masked = (im_template * cos_mask) + gray_c * np.ones(size_px_out)

        file_name = original_filenames[n].split(os.sep)[-1]

        file_parts = file_name.split(os.sep)[-1].split('.')[0].split('-')
        if file_parts[0].find('tex') == -1:
            type.append(1)
        else:
            type.append(2)

        position_y.append(stim_pos[0])
        position_x.append(stim_pos[1])
        sample.append(int(file_parts[3][file_parts[3].find('smp')+3:]))
        family_temp = int(file_parts[2][file_parts[2].find('im')+2:])
        family.append(np.where(familyNumOrder == family_temp)[0][0]+1)

        image_names[n] = 'aperture_' + file_name
        image_local_file_path[n] = save_dir + os.sep + image_names[n]

        imageio.imwrite(image_local_file_path[n], np.uint8(im_masked))

    stimuli = pd.DataFrame({'image_id': [str(n) for n in range(nStim)], 'degrees': [degrees] * nStim})

    stimuli['position_y'] = pd.Series(position_y)
    stimuli['position_x'] = pd.Series(position_x)
    stimuli['family'] = pd.Series(family)
    stimuli['sample'] = pd.Series(sample)
    stimuli['type'] = pd.Series(type)
    stimuli['image_file_name'] = pd.Series(image_names)
    stimuli['image_current_local_file_path'] = pd.Series(image_local_file_path)

    stimuli.to_csv(save_dir + os.sep + 'stimulus_set', index=False)


