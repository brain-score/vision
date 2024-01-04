
from collections import OrderedDict
from torch import nn
from .modules import VOneBlock, SequentialWithAllOutput, BottleneckVOne
from .back_ends import AlexNetBackEnd
from .params import generate_gabor_param
import numpy as np


def VOneNet(sf_corr=0.75, sf_max=6, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='Resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4,
            num_stochastic_copies=None, vone_outside_sequential=False):
    print('vone_outside_sequential', vone_outside_sequential)

    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size, num_stochastic_copies=num_stochastic_copies)

    if model_arch:
        bottleneck = BottleneckVOne(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            raise ValueError('%s BackEnd not implemented in this repository.'%model_arch.lower())
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            raise ValueError('%s BackEnd not implemented in this repository.'%model_arch.lower())

        if not vone_outside_sequential:
            model = SequentialWithAllOutput(OrderedDict([
                ('vone_block', vone_block),
                ('bottleneck', bottleneck),
                ('model', model_back_end),
            ]))
        else: # separate out the vone block (it has no learnable components) so we can treat is as preprocessing
            print('making separate vone block')
            model = SequentialWithAllOutput(OrderedDict([
                ('bottleneck', bottleneck),
                ('model', model_back_end),
            ]))
            vone_block_preproc = vone_block
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    if not vone_outside_sequential:
        return model
    else:
        return model, vone_block_preproc

