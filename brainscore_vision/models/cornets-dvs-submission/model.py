from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
from collections import OrderedDict
import math
import os
import requests
from torch.functional import F
import numpy as np
import scipy.stats as stats

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download():
    import sys, os
    # TAKE ID FROM SHAREABLE LINK
    file_id = "1X48v6JWs0dOa-YkWd8WwKHn2_sDtvd54"
    # DESTINATION FILE ON YOUR DISK
    destination = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights.pth")
    download_file_from_google_drive(file_id, destination)











def gabor_kernel(frequency,  sigma_x, sigma_y, theta=0, offset=0, ks=61):

    w = ks // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = torch.zeros(y.shape)
    g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= torch.cos(2 * np.pi * frequency * rotx + offset)

    return g


def sample_dist(hist, bins, ns, scale='linear'):
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample

class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)

def generate_gabor_param(features, seed=0, rand_flag=False, sf_corr=0, sf_max=9, sf_min=0):
    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        nx_bins = np.array([0.1, 10**0.2])
        nx_dist = np.array([1])

        ny_bins = np.array([0.1, 10**0.2])
        ny_dist = np.array([1])

        # sf_bins = np.array([0.5, 8])
        # sf_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1])

        sfmax_ind = np.where(sf_bins < sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()
    else:
        print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        ori_dist = ori_dist / ori_dist.sum()

        # Schiller 1976
        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
                                 [8.,  9.,  4.,  1.,  0.],
                                 [1.,  2., 19., 17.,  3.],
                                 [0.,  0.,  1.,  7.,  4.],
                                 [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([4,  4,  8, 25, 32, 26, 28, 12])

        sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()

    phase = sample_dist(phase_dist, phase_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)
    ori[ori < 0] = ori[ori < 0] + 180

    if rand_flag:
        sf = sample_dist(sf_dist, sf_bins, features, scale='log2')
        nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
        ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
    else:

        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10**nx

        ny_samp = np.random.rand(features)
        ny = np.zeros(features)
        for samp_ind, nx_samp in enumerate(nx):
            bin_id = np.argwhere(nx_bins < nx_samp)[-1]
            ny[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                             np.log10(ny_bins))
        ny = 10**ny

        sf = np.interp(samps_cdf[:,1], np.hstack(([0], sf_dist.cumsum())), np.log2(sf_bins))
        sf = 2**sf

    return sf, ori, phase, nx, ny




"""
Template module for a base model submission to brain-score
"""
class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x

class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1, in_channels = 3,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224):
        super().__init__()

        self.in_channels = in_channels

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None

def generate_vone_block(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            in_channels=3, simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=128, visual_degrees=8, ksize=25, stride=4):
    
    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    # gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
    #                 'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
    #                 'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    # arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}
    
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           in_channels=in_channels, simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    return vone_block, out_channels

class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels
    
    def ann_skipadd(self, x, skip):
        x += skip
        x = self.nonlin3(x)
        return x
    
    def snn_skipadd(self, x, skip):
        x = self.nonlin3(x)
        skip = self.nonlinskip(skip)
        x += skip
        return x
    
    def __init__(self, in_channels, out_channels, times=1, snn_type = False):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.nonlinskip = nn.ReLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU()

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))
            
        if snn_type:
            self.skipadd = self.snn_skipadd
        else:
            self.skipadd = self.ann_skipadd
    
    # def forward(self, inp):
    #     x = self.conv_input(inp)

    #     for t in range(self.times):
    #         if t == 0:
    #             skip = self.norm_skip(self.skip(x))
    #             self.conv2.stride = (2, 2)
    #         else:
    #             skip = x
    #             self.conv2.stride = (1, 1)

    #         x = self.conv1(x)
    #         x = getattr(self, f'norm1_{t}')(x)
    #         x = self.nonlin1(x)

    #         x = self.conv2(x)
    #         x = getattr(self, f'norm2_{t}')(x)
    #         x = self.nonlin2(x)

    #         x = self.conv3(x)
    #         x = getattr(self, f'norm3_{t}')(x)

    #         x += skip
    #         x = self.nonlin3(x)
    #         output = self.output(x)

    #     return output
    
    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            
            x = self.skipadd(x, skip)
            
            output = self.output(x)

        return output

def VOneCORnet_S(input_channels, output_range = 10):
    
    vone_block, out_channels = generate_vone_block(simple_channels=32,complex_channels=32,stride=2,in_channels=input_channels)
    model = nn.Sequential(OrderedDict([
        
        ('V1', vone_block),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, output_range)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['vonecornets-cifar10dvs']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'vonecornets-cifar10dvs'
    download()
    model = VOneCORnet_S(3,output_range=1000)
    dirname = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dirname, "weights.pth")
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=128)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(identifier='vonecornets-cifar10dvs', model=model, preprocessing=preprocessing)
    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 128
    return wrapper



def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    assert name == 'vonecornets-cifar10dvs'

    # returns the layers you want to consider
    return  ['V1', 'V2', 'V4', 'IT']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='vonecornets-cifar10dvs', activations_model=get_model("vonecornets-cifar10dvs"),
                        # specify layers to consider
                        layers=['V1', 'V2', 'V4', 'IT'])
