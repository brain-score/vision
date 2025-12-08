from collections import OrderedDict
from torch import nn
import numpy as np
from .modules import RetinaBlock, VOneBlock, Identity, EVBlock
from .params import get_tuned_params, generate_gabor_param,\
get_retinal_noise_fano, get_v1_noise_fano, get_v1_k_exc
from .backends import get_resnet, get_efficientnet, get_swin, get_vgg, get_alexnet, get_cornet_z

def EVNet(
    # RetinaBlock
    colors_p_cells=['r/g', 'g/r', 'b/y'], num_classes=1000, model_arch='resnet50',
    in_channels=3, p_channels=3, m_channels=0, image_size=224, visual_degrees=7,
    retinal_noise_mode=None, with_retinablock=True, with_voneblock=True,
    with_light_adapt=True, with_dog=True, with_contrast_norm=True, with_relu=False,
    # VOneBlock
    sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0, simple_channels=256, complex_channels=256,
    noise_mode=None, noise_scale=1, noise_level=0, ksize=31, stride=2, set_gabor_orientation=None,
    gabor_color_prob=None, k_exc=None, noise_fano=None, light_adapt_mode='weber', pool_dark=True, extract_representation=False, 
    lgn_to_v2=False
    ):

    retinablock_params, gabor_params, arch_params = None, None, None
    if with_retinablock:
        retinablock_params = get_tuned_params(
            p_channels, m_channels, colors_p_cells, ['w/b'], light_adapt_mode, visual_degrees, image_size
            )
        retinablock = RetinaBlock(
            **retinablock_params, in_channels=in_channels, m_channels=m_channels, p_channels=p_channels,
            fano_factor=get_retinal_noise_fano(), noise_mode=retinal_noise_mode, light_adapt_mode=light_adapt_mode,
            with_dog=with_dog, with_light_adapt=with_light_adapt, with_contrast_norm=with_contrast_norm, with_relu=with_relu,
            pool_dark=pool_dark
            )

    if with_voneblock:
        out_channels = simple_channels + complex_channels
        v1_in_channels = p_channels + m_channels if with_retinablock else in_channels
        k_exc = get_v1_k_exc(with_retinablock, image_size) if k_exc is None else k_exc
        noise_fano = get_v1_noise_fano((retinal_noise_mode is not None), image_size)  if noise_fano is None else noise_fano

        sf, theta, phase, nx, ny, color = generate_gabor_param(
                simple_channels, complex_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min,
                color_prob=gabor_color_prob, in_channels=v1_in_channels, set_orientation=set_gabor_orientation
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
        theta = theta / 180 * np.pi
        phase = phase / 180 * np.pi

        voneblock = VOneBlock(
            sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase, color=color, in_channels=v1_in_channels,
            k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level, fano_factor=noise_fano,
            simple_channels=simple_channels, complex_channels=complex_channels,
            ksize=ksize, stride=stride, input_size=image_size,
            )

    if model_arch:
        if model_arch.startswith('resnet'):
            assert model_arch[6:].isnumeric()
            backend = get_resnet(
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                layers=int(model_arch[6:]),
                backend=with_voneblock,
                extract_representation=extract_representation
                )
        if model_arch.startswith('efficientnet'):
            assert model_arch[-1].isnumeric()
            backend = get_efficientnet(
                b=int(model_arch[-1]),
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock,
                )
        if model_arch == 'alexnet':
            backend = get_alexnet(
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock
                )
        if model_arch == 'cornet_z':
            backend = get_cornet_z(
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock
                )
        if model_arch.startswith('swin'):
            backend = get_swin(
                model_arch,
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock,
                )
        if model_arch == 'vgg16':
            backend = get_vgg(
                layer_num=16,
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock
                )
        if model_arch == 'vgg19':
            backend = get_vgg(
                layer_num=19,
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock
                )

    model_dict = OrderedDict([])
    if lgn_to_v2:
        assert with_retinablock and with_voneblock and model_arch
        evblock = EVBlock(retinablock, voneblock, backend.in_channels, lgn_to_v2=True)
        model_dict.update({'evblock': evblock})
    else:
        if with_retinablock:
            model_dict.update({'retinablock': retinablock})
        if with_voneblock:
            model_dict.update({'voneblock': voneblock})
        if with_voneblock and len(model_arch) > 0:
            model_dict.update({'voneblock_bottleneck': nn.Conv2d(out_channels, backend.in_channels, 1, bias=False, groups=1)})
    
    if len(model_arch) > 0:
        model_dict.update({'model': backend})
    else:
        model_dict.update({'output': Identity()})

    model = nn.Sequential(model_dict)

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.retinablock_params = retinablock_params
    model.gabor_params = gabor_params
    model.arch_params = arch_params
    model.is_stochastic = False if (retinal_noise_mode is None and noise_mode is None) else True

    return model
