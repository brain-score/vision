import torch
from collections import OrderedDict
from torch import nn
import torchvision
import numpy as np
from .modules import RetinaBlock, VOneBlock, Identity
from .params import get_tuned_params, generate_gabor_param,\
get_retinal_noise_fano, get_v1_noise_fano, get_v1_k_exc
from .backends import get_resnet, get_vgg


def EVNet(
    # RetinaBlock
    colors_p_cells=['r/g', 'g/r', 'b/y'], num_classes=1000, model_arch='resnet50',
    in_channels=3, p_channels=3, m_channels=0, image_size=224, visual_degrees=7,
    retinal_noise_mode=None, with_retinablock=True, with_voneblock=True,
    with_light_adapt=True, with_dog=True, with_contrast_norm=True, with_relu=False,
    # VOneBlock
    sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0, simple_channels=256, complex_channels=256,
    noise_mode=None, noise_scale=1, noise_level=0, ksize=31, stride=2, set_gabor_orientation=None,
    gabor_color_prob=None
    ):

    retinablock_params, gabor_params, arch_params = None, None, None
    if with_retinablock:
        retinablock_params = get_tuned_params(
            p_channels, m_channels, colors_p_cells, ['w/b'], visual_degrees, image_size
            )
        retinablock = RetinaBlock(
            **retinablock_params, in_channels=in_channels, m_channels=m_channels, p_channels=p_channels,
            fano_factor=get_retinal_noise_fano(), noise_mode=retinal_noise_mode,
            with_dog=with_dog, with_light_adapt=with_light_adapt, with_contrast_norm=with_contrast_norm, with_relu=with_relu
            )

    if with_voneblock:
        out_channels = simple_channels + complex_channels
        k_exc = get_v1_k_exc(with_retinablock=with_retinablock, gabor_color_prob=gabor_color_prob)
        noise_fano = get_v1_noise_fano((retinal_noise_mode is not None), image_size)

        sf, theta, phase, nx, ny, color = generate_gabor_param(
                simple_channels, complex_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min,
                color_prob=gabor_color_prob, in_channels=(p_channels+m_channels), set_orientation=set_gabor_orientation
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
                )
        if model_arch == 'vgg16':
            backend, backend_in_channels = get_vgg(
                in_channels=(p_channels+m_channels if with_retinablock else in_channels),
                num_classes=num_classes,
                layers=16,
                backend=with_voneblock,
                tiny=(image_size<=64),
                )
        if model_arch == 'efficientnet_b0':
            #backend = torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            backend = torchvision.models.efficientnet_b0()
            for module in backend.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = .99
            backend.features[0][0].stride = (4,4)
            backend.features[0][0].kernel_size = (7,7)
            backend.features[0][0].padding = (3,3)
            backend.classifier[-1] = nn.Linear(backend.classifier[-1].in_features, num_classes)
            if with_voneblock:
                backend_in_channels = backend.features[1][0].block[0][0].in_channels
                backend.features = backend.features[1:]  # Remove first block from EfficientNet-B0
            else:
                if (p_channels + m_channels) > 3:
                    conv1 = nn.Conv2d(
                        in_channels=(p_channels + m_channels),
                        out_channels=backend.features[0][0].out_channels,
                        kernel_size=backend.features[0][0].kernel_size,
                        stride=backend.features[0][0].stride,
                        padding=backend.features[0][0].padding,
                        bias=backend.features[0][0].bias
                        )
                    weight = torch.zeros_like(conv1.weight.data)
                    weight[:, :3, :, :] = backend.features[0][0].weight.data
                    nn.init.kaiming_normal_(weight[:, 3:, :, :])
                    conv1.weight.data = weight
                    backend.features[0][0]= conv1

    model_dict = OrderedDict([])
    if with_retinablock:
        model_dict.update({'retinablock': retinablock})
    if with_voneblock:
        model_dict.update({'voneblock': voneblock})
    if with_voneblock and model_arch:
        model_dict.update({'voneblock_bottleneck': nn.Conv2d(out_channels, backend.in_channels, 1, bias=False, groups=1)})
    if model_arch:
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
