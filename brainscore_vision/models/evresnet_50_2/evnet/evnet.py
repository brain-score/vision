from collections import OrderedDict
from torch import nn
import numpy as np
from .modules import RetinaBlock, Bottleneck, VOneBlock, Identity
from .params import get_dog_params, get_div_norm_params, generate_gabor_param
from .backends import get_resnet_backend, get_vgg_backend

evnet_params = {
        'base': {
            'with_retinablock': False,
            'with_voneblock': False
            },
        'vonenet': {
            'with_retinablock': False,
            'with_voneblock': True
            },
        'retinanet': {
            'with_retinablock': True,
            'relative_size_la': 4,
            'colors_p_cells': ['r/g', 'g/r', 'b/y'],
            'contrast_norm': True,
            'p_channels': 3,
            'm_channels': 1,
            'linear_p_cells': True,
            'with_voneblock': False
        },
        'evnet': {
            'with_retinablock': True,
            'relative_size_la': 4,
            'colors_p_cells': ['r/g', 'g/r', 'b/y'],
            'contrast_norm': True,
            'p_channels': 3,
            'm_channels': 1,
            'linear_p_cells': True,
            'with_voneblock': True
        }
    }

def EVNet(
    # RetinaBlock
    colors_p_cells=['r/g', 'g/r', 'b/y'], relative_size_la=4, dog_across_channels=True, contrast_norm=True,
    with_retinablock=True, with_voneblock=True, sampling='median', num_classes=200, model_arch='resnet18',
    in_channels=3, p_channels=3, m_channels=0, image_size=64, visual_degrees=2,
    linear_p_cells=False,
    # VOneBlock
    sf_corr=0.75, sf_max=11.5, sf_min=0, rand_param=False, gabor_seed=0, simple_channels=256, complex_channels=256,
    noise_mode=None, noise_scale=1, noise_level=1, k_exc=1, ksize=25, stride=2, set_gabor_orientation=None
    ):

    dog_params, div_norm_params, gabor_params, arch_params = None, None, None, None
    if with_retinablock:
        dog_params = get_dog_params(
            features=p_channels, colors=colors_p_cells, cell_type='p',
            image_size=image_size, visual_degrees=visual_degrees, sampling=sampling
            )
        dog_params.update(get_dog_params(
        features=m_channels, colors=['w/b'], cell_type='m',
        image_size=image_size, visual_degrees=visual_degrees, sampling=sampling
        ))

        div_norm_params = get_div_norm_params(
            relative_size_la=relative_size_la,
            image_size=image_size, visual_degrees=visual_degrees
            )
        retinablock = RetinaBlock(
            **dog_params, **div_norm_params,
            in_channels=in_channels, p_channels=p_channels, m_channels=m_channels,
            linear_p_cells=linear_p_cells, contrast_norm=contrast_norm, dog_across_channels=dog_across_channels
            )

    if with_voneblock:
        out_channels = simple_channels + complex_channels
        sf, theta, phase, nx, ny, color = generate_gabor_param(
            simple_channels, complex_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min,
            in_channels=(p_channels+m_channels), set_orientation=set_gabor_orientation
            )
        gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                        'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                        'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy(), 'color': color.copy()}

        arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}

        # Conversions
        ppd = image_size / visual_degrees
        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta/180 * np.pi
        phase = phase / 180 * np.pi
    
        voneblock = VOneBlock(
            sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase, color=color, in_channels=(p_channels + m_channels),
            k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
            simple_channels=simple_channels, complex_channels=complex_channels,
            ksize=ksize, stride=stride, input_size=image_size
            )

    if model_arch:
        if model_arch == 'resnet18':
            backend, backend_in_channels = get_resnet_backend(
                p_channels=p_channels,
                m_channels=m_channels,
                num_classes=num_classes,
                with_voneblock=with_voneblock,
                tiny=(image_size==64),
                layers=18
                )
        if model_arch == 'resnet50':
            backend, backend_in_channels = get_resnet_backend(
                p_channels=p_channels,
                m_channels=m_channels,
                num_classes=num_classes,
                with_voneblock=with_voneblock,
                tiny=(image_size==64),
                layers=50
                )
        if model_arch == 'vgg16':
            backend, backend_in_channels = get_vgg_backend(
                p_channels=p_channels,
                m_channels=m_channels,
                num_classes=num_classes,
                with_voneblock=with_voneblock,
                tiny=(image_size==64),
                layers=16
                )

    model_dict = OrderedDict([])
    if with_retinablock:
        model_dict.update({'retinablock': retinablock})
    if with_voneblock:
        model_dict.update({'voneblock': voneblock})
        model_dict.update({'voneblock_bottleneck': Bottleneck(out_channels, backend_in_channels, 1)})
    if model_arch:
        model_dict.update({'model': backend})
    else:
        model_dict.update({'output': Identity()})

    model = nn.Sequential(model_dict)

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.dog_params = dog_params
    model.div_norm_params = div_norm_params
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model
