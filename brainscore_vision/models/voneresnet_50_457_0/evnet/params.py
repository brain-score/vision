import torch
import numpy as np
import scipy.stats as stats
from .utils import sample_dist
from typing import Literal

image_size = 224
visual_degrees = 7
ppd = (image_size/visual_degrees)
kernel_size = {'p': 21, 'm': 65}  # 95% of Gaussian coverage


# Receptive fields of P and M ganglion cells across the primate retina (Kroner and Kaplan, 1994)
# https://www.sciencedirect.com/science/article/pii/0042698994E0066T

# P Cells (eccentricity range of 0-5)
P_cell_params = {
    'med_rc': 0.03, # Center radius
    'med_kc': 325.2, # Center peak sensitivity
    'med_rs': 0.18,  # Surround radius
    'med_ks': 4.4, # Surround peak sensitivity
    'c_kc': 0.391,  # Center peak sensitivity vs. radius regression
    'c_ks': 0.128,  # Surround peak sensitivity vs. radius regression
}

# M Cells (eccentricity range of 0-10)
M_cell_params = {
    'med_rc': 0.10, # Center radius
    'med_kc': 148.0, # Center peak sensitivity
    'med_rs': 0.72, # Surround radius
    'med_ks': 1.1, # Surround peak sensitivity
}

# From Bayesian Optimization
P_cell_params = {
    'rc_dog': 1.2375,
    'rs_dog': 7.1528,
    'kernel_dog': 21,
    'ratio_dog': 0.017388,
    'radius_la': np.inf,
    'kernel_la': 1,
    'exp_la': 0.300,
    'radius_cn': 13.1507,
    'kernel_cn': 35,
    'c50_cn': 0.0293,
    'exp_cn': 0.4681,
    'k_exc': 11.6264
}

M_cell_params = {
    'rc_dog': 2.3546,
    'rs_dog': 19.2944,
    'kernel_dog': 51,
    'ratio_dog': 0.008129,
    'radius_la': np.inf,
    'kernel_la': 1,
    'exp_la': 0.8689,
    'radius_cn': 28.1936,
    'kernel_cn': 73,
    'c50_cn': 0.01,
    'exp_cn': 0.499,
    'k_exc': 5.8508
}

def get_kernel_size(radius, energy=.9):
    if radius==np.inf:
        return 1
    assert 0 < energy < 1
    sigma = radius/np.sqrt(2)
    r_min = np.sqrt(-2*np.log(1-energy)) * sigma
    return int(2*np.ceil(r_min)+1)

def get_dog_params(
    features:int, colors:list[Literal['r/g', 'g/r', 'b/y', 'w/b']]=['r/g', 'g/r', 'b/y'],
    polarity:list[Literal[0, 1]]=None,
    cell_type:Literal['p', 'm']='p', image_size:int=image_size, visual_degrees:int=visual_degrees
    ) -> dict:
    """Generates DoG parameters for RetinaBlock with more than 3 channels.
    Number of channels = number of features * 3 color options (R/G, G/R, B/Y).
    Only generates ON-center cells.

    Args:
        features (int): _description_
        binning (bool, optional): whether to use discrete binning while sampling values. Defaults to True.
        image_size (int, optional): model image size. Defaults to image_size.
        visual_degrees (int, optional): visual degrees of the model FoV. Defaults to visual_degrees.

    Returns:
        dict: dictionary with center and surround radii, opponency tensor and DoG kernel size
    """
    
    if not features:
        return {
                f'rc_{cell_type}_cell': torch.tensor([]),
                f'rs_{cell_type}_cell': torch.tensor([]),
                f'opponency_{cell_type}_cell': torch.tensor([]),
                f'kernel_{cell_type}_cell': torch.tensor([])
                }

    assert cell_type in ['p', 'm']

    cell_params = M_cell_params if cell_type=='m' else P_cell_params
    color_mapping = {
            'r/g': np.array([[1,0,0],[0,-1,0]], dtype=np.float32),  # R+/G- (center/surround)
            'g/r': np.array([[0,1,0],[-1,0,0]], dtype=np.float32),  # G+/R-
            'b/y': np.array([[0,0,1],[-.5,-.5,0]], dtype=np.float32),  # B+/Y-
            'w/b': np.array([[1/3]*3,[-1/3]*3], dtype=np.float32)  # ON/OFF
        }
    
    assert features % len(colors) == 0
    
    # Use median values from distributions (deterministic)
    assert features == len(colors)
    rc = np.ones((features,), dtype=np.float32) * cell_params['med_rc']
    rs = np.ones((features,), dtype=np.float32) * cell_params['med_rs']
    kc = np.ones((features,), dtype=np.float32) * cell_params['med_kc']
    ks = np.ones((features,), dtype=np.float32) * cell_params['med_ks']

    opponency = np.concatenate([
        np.repeat(color_mapping[c][None, ...], features // len(colors), axis=0)
        for c in colors
        ])

    # Conversions
    ppd = image_size / visual_degrees  # pixels per FOV degree
    rc = torch.from_numpy(rc * ppd)
    rs = torch.from_numpy(rs * ppd)
    kc = torch.from_numpy(kc / ppd ** 2)
    ks = torch.from_numpy(ks / ppd ** 2)
    opponency = torch.from_numpy(opponency)

    opponency[:,1] *= torch.unsqueeze(ks[:]/kc[:], 1)
    
    if polarity:
        assert len(polarity) == opponency.size(0)
        opponency *= torch.tensor(polarity)[..., None, None]
    
    params = {
        f'rc_{cell_type}_cell': rc,
        f'rs_{cell_type}_cell': rs,
        f'opponency_{cell_type}_cell': opponency,
        f'kernel_{cell_type}_cell': kernel_size[cell_type]
    }

    return params


def get_div_norm_params(
    relative_size_la, kernel_la=None, 
    image_size=image_size, visual_degrees=visual_degrees
    ) -> dict:

    # Conversions
    ppd = image_size / visual_degrees  # pixels per FOV degree
    radius_la = P_cell_params['med_rs'] * relative_size_la * ppd
    radius_cn = 2 * P_cell_params['med_rc'] * ppd
    c50 = .3

    if not kernel_la and radius_la < np.inf:
        kernel_la = int(radius_la*2) + int(int(radius_la*2)%2==0)
    
    params = {
        'kernel_la': kernel_la,
        'radius_la': radius_la,
        'kernel_cn': kernel_size['p'],
        'radius_cn': radius_cn,
        'c50': c50
    }

    return params


def get_tuned_params(
    features_p, features_m, colors_p=['r/g', 'g/r', 'b/y'], colors_m=['w/b'],
    visual_degrees=visual_degrees, image_size=image_size
    ):
    assert features_p % len(colors_p) == 0 and features_m % len(colors_m) == 0

    res_factor = (image_size/visual_degrees) / (224/7)
    
    color_mapping = {
        'r/g': np.array([[1,0,0],[0,-1,0]], dtype=np.float32),  # R+/G- (center/surround)
        'g/r': np.array([[0,1,0],[-1,0,0]], dtype=np.float32),  # G+/R-
        'b/y': np.array([[0,0,1],[-.5,-.5,0]], dtype=np.float32),  # B+/Y-
        'w/b': np.array([[1/3]*3,[-1/3]*3], dtype=np.float32)  # ON/OFF
    }

    num_conv_p = bool(features_p)
    num_conv_m = bool(features_m)

    # DoG Radii
    rc = np.array([P_cell_params['rc_dog']]*num_conv_p+[M_cell_params['rc_dog']]*num_conv_m, dtype=np.float32)
    rs = np.array([P_cell_params['rs_dog']]*num_conv_p+[M_cell_params['rs_dog']]*num_conv_m, dtype=np.float32)
    rc *= res_factor
    rs *= res_factor

    # DoG Opponency
    opponency = np.zeros((features_p+features_m, 2, 3), dtype=np.float32)
    opponency[:features_p] = np.concatenate([color_mapping[c][None, ...] for c in colors_p])
    opponency[:features_p, 1] *= P_cell_params['ratio_dog']
    opponency[features_p:] = np.concatenate([color_mapping[c][None, ...] for c in colors_m])
    opponency[features_p:, 1] *= M_cell_params['ratio_dog']

    kernel_dog = np.array([get_kernel_size(r, energy=.8) for r in rs.tolist()], dtype=np.int16)
    
    # Light Adaptation
    exp_la = np.array([P_cell_params['exp_la']]*num_conv_p+[M_cell_params['exp_la']]*num_conv_m, dtype=np.float32)
    radius_la = np.array([P_cell_params['radius_la']]*num_conv_p+[M_cell_params['radius_la']]*num_conv_m, dtype=np.float32)
    radius_la *= res_factor
    kernel_la = np.array([get_kernel_size(r, energy=.8) for r in radius_la.tolist()], dtype=np.int16)
    
    # Contrast Normalization
    c50_cn = np.array([P_cell_params['c50_cn']]*num_conv_p+[M_cell_params['c50_cn']]*num_conv_m, dtype=np.float32)
    exp_cn = np.array([P_cell_params['exp_cn']]*num_conv_p+[M_cell_params['exp_cn']]*num_conv_m, dtype=np.float32)
    radius_cn = np.array([P_cell_params['radius_cn']]*num_conv_p+[M_cell_params['radius_cn']]*num_conv_m, dtype=np.float32)
    radius_cn *= res_factor
    kernel_cn = np.array([get_kernel_size(r, energy=.8) for r in radius_cn.tolist()], dtype=np.int16)

    k_exc = np.array([P_cell_params['k_exc']]*features_p+[M_cell_params['k_exc']]*features_m)

    ks_max = image_size * 2 - 1
    return {
        'rc_dog': rc,
        'rs_dog': rs,
        'opponency_dog': opponency,
        'kernel_dog': np.minimum(kernel_dog, ks_max),
        'kernel_la': np.minimum(kernel_la, ks_max),
        'radius_la': radius_la,
        'exp_la': exp_la,
        'kernel_cn': np.minimum(kernel_cn, ks_max),
        'radius_cn': radius_cn,
        'c50_cn': c50_cn,
        'exp_cn': exp_cn,
        'k_exc': k_exc
    }

def get_v1_k_exc(with_retinablock, gabor_color_prob):
    if not with_retinablock: return 15.412
    if gabor_color_prob==[.25,.25,.25,.25]: return 3.304
    if gabor_color_prob==[.3,.3,.3,.1]: return 3.206
    raise ValueError(f'gabor_color_prob is {gabor_color_prob}.')

def get_retinal_noise_fano():
    return .4

def get_v1_noise_fano(with_retinal_noise, image_size):
    return (.98 if image_size==224 else .955) if with_retinal_noise else 1

def get_grating_params(
    sf, angle=0, phase=0, contrast=1, radius=.5,
    image_size=image_size, visual_degrees=visual_degrees
    ) -> dict:
    ppd = image_size / visual_degrees  # pixels per FOV degree
    params = {
        'size': image_size,
        'radius': radius * ppd,
        'sf': sf/ppd,
        'theta': angle,
        'phase': phase,
        'contrast': contrast
    }
    return params


def generate_gabor_param(
    n_sc, n_cc, seed=0, rand_flag=False, sf_corr=0.75,
    sf_max=11.5, sf_min=0, diff_n=False, dnstd=0.22,
    # Additional parameters
    in_channels=3, color_prob=None,
    set_orientation=None, verbose=False
    ):
    
    features = n_sc + n_cc

    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        if verbose: print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        # nx_bins = np.array([0.1, 10**0.2])
        nx_bins = np.array([0.1, 10**0])
        nx_dist = np.array([1])

        # ny_bins = np.array([0.1, 10**0.2]
        ny_bins = np.array([0.1, 10**0])
        ny_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        sf_s_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])
        sf_c_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])

    else:
        if verbose: print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        # ori_dist = np.array([110, 83, 100, 92])
        ori_dist = ori_dist / ori_dist.sum()

        # Ringach 2002b
        # nx_bins = np.logspace(-1, 0.2, 6, base=10)
        # ny_bins = np.logspace(-1, 0.2, 6, base=10)
        nx_bins = np.logspace(-1, 0., 5, base=10)
        ny_bins = np.logspace(-1, 0., 5, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.],
                                 [8.,  9.,  4.,  1.],
                                 [1.,  2., 19., 17.],
                                 [0.,  0.,  1.,  7.]])
        # n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
        #                          [8.,  9.,  4.,  1.,  0.],
        #                          [1.,  2., 19., 17.,  3.],
        #                          [0.,  0.,  1.,  7.,  4.],
        #                          [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        # foveal only
        sf_s_dist = np.array([4, 4, 8, 25, 33, 26, 28, 12, 8])
        sf_c_dist = np.array([0, 0, 9, 9, 7, 10, 23, 12, 14])
        # foveal + parafoveal
        # sf_s_dist = np.array([8, 14, 20, 43, 40, 44, 31, 16, 8])
        # sf_c_dist = np.array([2, 1, 11, 14, 22, 23, 32, 15, 16])


    phase = sample_dist(phase_dist, phase_bins, features)
    if set_orientation or set_orientation == 0:
        ori = np.ones((features,)) * set_orientation
    else:
        ori = sample_dist(ori_dist, ori_bins, features)

    # ori[ori < 0] = ori[ori < 0] + 180
    
    sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
    sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

    sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
    sf_s_dist = sf_s_dist[sfmin_ind:sfmax_ind]
    sf_c_dist = sf_c_dist[sfmin_ind:sfmax_ind]

    sf_s_dist = sf_s_dist / sf_s_dist.sum()
    sf_c_dist = sf_c_dist / sf_c_dist.sum()

    cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

    if rand_flag:   # Uniform
        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10**nx

        if diff_n: 
            ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
        else:
            ny = 10**(np.random.normal(np.log10(nx), dnstd))
            ny[ny<0.1] = 0.1
            ny[ny>1] = 1
            # ny = nx

        sf = np.interp(samps_cdf[:,1], np.hstack(([0], sf_s_dist.cumsum())), np.log2(sf_bins))
        sf = 2**sf

        # if n_sc > 0:
        #     sf_s = sample_dist(sf_s_dist, sf_bins, n_sc, scale='log2')
        # else:
        #     sf_s = np.array([])
        # if n_cc > 0:
        #     sf_c = sample_dist(sf_c_dist, sf_bins, n_cc, scale='log2')
        # else:
        #     sf_c = np.array([])
        # sf = np.concatenate((sf_s, sf_c))

        # nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
    else:   # Biological

        if n_sc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_sc)
            samps_cdf = stats.norm.cdf(samps)

            nx_s = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_s = 10**nx_s

            ny_samp = np.random.rand(n_sc)
            ny_s = np.zeros(n_sc)
            for samp_ind, nx_samp in enumerate(nx_s):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_s[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_s = 10**ny_s

            sf_s = np.interp(samps_cdf[:,1], np.hstack(([0], sf_s_dist.cumsum())), np.log2(sf_bins))
            sf_s = 2**sf_s
        else:
            nx_s = np.array([])
            ny_s = np.array([])
            sf_s = np.array([])

        if n_cc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_cc)
            samps_cdf = stats.norm.cdf(samps)

            nx_c = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_c = 10**nx_c

            ny_samp = np.random.rand(n_cc)
            ny_c = np.zeros(n_cc)
            for samp_ind, nx_samp in enumerate(nx_c):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_c[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_c = 10**ny_c

            sf_c = np.interp(samps_cdf[:,1], np.hstack(([0], sf_c_dist.cumsum())), np.log2(sf_bins))
            sf_c = 2**sf_c
        else:
            nx_c = np.array([])
            ny_c = np.array([])
            sf_c = np.array([])

        nx = np.concatenate((nx_s, nx_c))
        ny = np.concatenate((ny_s, ny_c))
        sf = np.concatenate((sf_s, sf_c))

    # Generate an array of size 'features', with values either 0,1,2, (pseudo)randomly set
    if color_prob:
        counts = [round(p*features) for p in color_prob]
        counts[2] += 512 - sum(counts)  # Y/B gets one less
        color = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        np.random.shuffle(color)
    else:
        color = np.random.randint(low=0, high=in_channels, size=features, dtype=np.int8)

    return sf, ori, phase, nx, ny, color
