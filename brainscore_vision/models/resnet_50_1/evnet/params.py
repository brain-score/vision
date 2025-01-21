import torch
import numpy as np
import scipy.stats as stats
from .utils import sample_dist
from typing import Literal

image_size = 224
visual_degrees = 7
kernel_size = {'p': 21, 'm': 65}  # 95% of Gaussian coverage


# Receptive fields of P and M ganglion cells across the primate retina (Kroner and Kaplan, 1994)
# https://www.sciencedirect.com/science/article/pii/0042698994E0066T

# P Cells (eccentricity range of 0-5)
P_cell_params = {
    'med_rc': 0.03, 'iqr_rc': 0.01,  # Center radius
    'med_kc': 325.2, 'iqr_kc': 302,  # Center peak sensitivity
    'med_rs': 0.18, 'iqr_rs': 0.07,  # Surround radius
    'med_ks': 4.4, 'iqr_ks': 4.6,  # Surround peak sensitivity
    'c_kc': 0.391, 'm_kc': -1.850,  # Center peak sensitivity vs. radius regression
    'c_ks': 0.128, 'm_ks': -2.147,  # Surround peak sensitivity vs. radius regression
}

# M Cells (eccentricity range of 0-10)
M_cell_params = {
    'med_rc': 0.10, 'iqr_rc': 0.02,  # Center radius
    'med_kc': 148.0, 'iqr_kc': 122.4,  # Center peak sensitivity
    'med_rs': 0.72, 'iqr_rs': 0.23,  # Surround radius
    'med_ks': 1.1, 'iqr_ks': 0.8,  # Surround peak sensitivity
}


def get_dog_params(
    features:int, sampling:Literal['median', 'binning', 'uniform', 'lognormal']='median',
    colors:list[Literal['r/g', 'g/r', 'b/y', 'w/b']]=['r/g', 'g/r', 'b/y'],
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
    min_rc = cell_params['med_rc'] - cell_params['iqr_rc']
    max_rc = cell_params['med_rc'] + cell_params['iqr_rc']
    min_rs = cell_params['med_rs'] - cell_params['iqr_rs']
    max_rs = cell_params['med_rs'] + cell_params['iqr_rs']

    color_mapping = {
            'r/g': np.array([[1,0,0],[0,-1,0]], dtype=np.float16),  # R+/G- (center/surround)
            'g/r': np.array([[0,1,0],[-1,0,0]], dtype=np.float16),  # G+/R-
            'b/y': np.array([[0,0,1],[-.5,-.5,0]], dtype=np.float16),  # B+/Y-
            'w/b': np.array([[1/3]*3,[-1/3]*3], dtype=np.float16)  # ON/OFF
        }
    
    assert features % len(colors) == 0
    
    if sampling=='median':
        # Use median values from distributions (deterministic)
        assert features == len(colors)
        rc = np.ones((features,), dtype=np.float16) * cell_params['med_rc']
        rs = np.ones((features,), dtype=np.float16) * cell_params['med_rs']
        kc = np.ones((features,), dtype=np.float16) * cell_params['med_kc']
        ks = np.ones((features,), dtype=np.float16) * cell_params['med_ks']
    elif sampling=='binning':
        # Assume uniform joint distribution of rc and rs with discrete binning (deterministic)
        assert int(np.sqrt(features//len(colors)))==np.sqrt(features//len(colors))
        edges_rc = np.linspace(min_rc, max_rc, int(np.sqrt(features // len(colors))) + 1)
        edges_rs = np.linspace(min_rs, max_rs, int(np.sqrt(features // len(colors))) + 1)
        centers_rc = (edges_rc[:-1] + edges_rc[1:]) / 2
        centers_rs = (edges_rs[:-1] + edges_rs[1:]) / 2
        rc = np.repeat(centers_rc, int(np.sqrt(features // len(colors))))
        rs = np.tile(centers_rs, int(np.sqrt(features // len(colors))))
    elif sampling=='uniform':
        # Assume uniform disjoint distribution of rc and rs without binning (stochastic)
        rc = np.random.uniform(min_rc, max_rc, features // len(colors))
        rs = np.random.uniform(min_rs, max_rs, features // len(colors))
    elif sampling=='lognormal':
        # Assume lognormal disjoint distribution of rc and rs (stochastic)
        std_rc = (np.log(cell_params['med_rc'] - (cell_params['iqr_rc']/2)) - np.log(cell_params['med_rc'])) / stats.norm.ppf(.25)
        std_rs = (np.log(cell_params['med_rs'] - (cell_params['iqr_rs']/2)) - np.log(cell_params['med_rs'])) / stats.norm.ppf(.25)
        rc = np.random.lognormal(np.log(cell_params['med_rc']), std_rc, features // len(colors))
        rs = np.random.lognormal(np.log(cell_params['med_rs']), std_rs, features // len(colors))
    
    if sampling != 'median':
        assert cell_type == 'p'
        rc = np.tile(rc, len(colors))
        rs = np.tile(rs, len(colors))
        kc = cell_params['c_kc'] * rc ** cell_params['m_kc']
        ks = cell_params['c_ks'] * rs ** cell_params['m_ks']

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
    in_channels=3, set_orientation=None
    ):
    
    features = n_sc + n_cc

    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        nx_bins = np.array([0.1, 10**0])
        nx_dist = np.array([1])

        ny_bins = np.array([0.1, 10**0])
        ny_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        sf_s_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])
        sf_c_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])

    else:
        print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        ori_dist = ori_dist / ori_dist.sum()

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0., 5, base=10)
        ny_bins = np.logspace(-1, 0., 5, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.],
                                 [8.,  9.,  4.,  1.],
                                 [1.,  2., 19., 17.],
                                 [0.,  0.,  1.,  7.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        # foveal only
        sf_s_dist = np.array([4, 4, 8, 25, 33, 26, 28, 12, 8])
        sf_c_dist = np.array([0, 0, 9, 9, 7, 10, 23, 12, 14])

    phase = sample_dist(phase_dist, phase_bins, features)

    if set_orientation or set_orientation == 0:
        ori = np.ones((features,)) * set_orientation
    else:
        ori = sample_dist(ori_dist, ori_bins, features)
    
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

    color = np.random.randint(low=0, high=in_channels, size=features, dtype=np.int8)

    return sf, ori, phase, nx, ny, color
