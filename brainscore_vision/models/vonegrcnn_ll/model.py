## Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
import torch
#from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
#from candidate_models import s3 

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math
import os
import scipy.stats as stats
from collections import OrderedDict

#from vonenet import get_model
#from vonenet.vonenet import VOneNet

os.environ["CUDA_VISIBLE_DEVICES"]="0"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


   
import logging
import os
import sys


#VARIABLES
model_identifier = ' vone_grcnn_ll'


###### DEFINE ELEMENTS OF THE MODELS

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

class Identity(nn.Module):
    def forward(self, x):
        return x

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

class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224):
        super().__init__()

        self.in_channels = 3

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True) # inplace=True
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=224, visual_degrees=8, ksize=25, stride=4):


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
                           ksize=ksize, stride=stride, input_size=image_size)

    bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
    nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

    return vone_block, bottleneck

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32, groups=32):

        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList() 
        for i in range(M):

            conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=groups,bias=False)
            init.kaiming_normal_(conv1.weight)
            self.conv.append(nn.Sequential(conv1,
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) 
        conv_fc = nn.Conv2d(out_channels,d,1,bias=False)
        init.normal_(conv_fc.weight, std=0.01)
        self.fc1=nn.Sequential(conv_fc,
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        init.normal_(self.fc2.weight, std=0.01)
        self.softmax=nn.Softmax(dim=1) 

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        for i,conv in enumerate(self.conv):
            output.append(conv(input))
        U=reduce(lambda x,y:x+y,output) 
        s=self.global_pool(U)
        z=self.fc1(s)  
        a_b=self.fc2(z) 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V=list(map(lambda x,y:x*y,output,a_b)) 
        V=reduce(lambda x,y:x+y,V) 
        return V

class GRCL(nn.Module):
  def __init__(self, inplanes, planes, downsample=True, iter = 3, SKconv=True, expansion=2):
    super(GRCL, self).__init__()

    self.iter = iter
    self.expansion = expansion
    # feed-forward part
    self.add_module('bn_f', nn.BatchNorm2d(inplanes))
    self.add_module('relu_f', nn.ReLU(inplace=True))
    conv_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=3, stride=1, padding=1, bias=False, groups=32)
    init.kaiming_normal_(conv_f.weight)
    self.add_module('conv_f', conv_f)
    
    self.add_module('bn_g_f', nn.BatchNorm2d(inplanes))
    self.add_module('relu_g_f', nn.ReLU(inplace=True))
    conv_g_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=True, groups=32)
    init.normal_(conv_g_f.weight, std=0.01)
    self.add_module('conv_g_f', conv_g_f)
    self.conv_g_r = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=False, groups=32)
    self.add_module('sig', nn.Sigmoid())

    # recurrent part
    for i in range(0, self.iter):
     layers = []
     layers_g_bn = []
    
     layers.append(nn.BatchNorm2d(planes*self.expansion))
     layers.append(nn.ReLU(inplace=True))
     conv_1 = nn.Conv2d(int(planes*self.expansion), planes, kernel_size=1, stride=1, padding=0, bias=False)
     init.kaiming_normal_(conv_1.weight)
     layers.append(conv_1)

     layers.append(nn.BatchNorm2d(planes))
     layers.append(nn.ReLU(inplace=True))

     if SKconv:
       layers.append(SKConv(planes, planes))
     else:
       layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
       layers.append(nn.BatchNorm2d(planes))
       layers.append(nn.ReLU(inplace=True))

     conv_2 = nn.Conv2d(planes, int(planes*self.expansion), kernel_size=1, stride=1, padding=0, bias=False)   
     init.kaiming_normal_(conv_2.weight)
     layers.append(conv_2)
     layers_g_bn.append(nn.BatchNorm2d(int(planes*self.expansion)))

     layers_g_bn.append(nn.ReLU(inplace=True)) 

     self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
     self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))

    self.downsample = downsample
    if self.downsample:
       self.add_module('d_bn', nn.BatchNorm2d(planes * self.expansion))
       self.add_module('d_relu', nn.ReLU(inplace=True))
       d_conv = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv.weight)
       self.add_module('d_conv', d_conv)
       self.add_module('d_ave', nn.AvgPool2d((2, 2), stride=2))
  
       self.add_module('d_bn_1', nn.BatchNorm2d(planes * self.expansion))
       self.add_module('d_relu_1', nn.ReLU(inplace=True))
       d_conv_1 = nn.Conv2d(int(planes* self.expansion), planes, kernel_size=1, stride=1, padding=0,
       bias=False)
       init.kaiming_normal_(d_conv_1.weight)
       self.add_module('d_conv_1', d_conv_1)

       self.add_module('d_bn_3', nn.BatchNorm2d(planes))
       self.add_module('d_relu_3', nn.ReLU(inplace=True))
       
       if SKconv:
         d_conv_3 = SKConv(planes, planes, stride=2)
         self.add_module('d_conv_3', d_conv_3)
       else:
         d_conv_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
         init.kaiming_normal_(d_conv_3.weight)
         self.add_module('d_conv_3', d_conv_3)

       d_conv_1e = nn.Conv2d(planes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv_1e.weight)
       self.add_module('d_conv_1e', d_conv_1e)

  def forward(self, x):
    # feed-forward
    x_bn = self.bn_f(x)
    x_act = self.relu_f(x_bn)
    x_s = self.conv_f(x_act)
    
    x_g_bn = self.bn_g_f(x)
    x_g_act = self.relu_g_f(x_g_bn)
    x_g_s = self.conv_g_f(x_g_act)

    # recurrent 
    for i in range(0, self.iter):
       x_g_r = self.conv_g_r(self.__dict__['_modules']["iter_g_%s" % str(i+1)](x_s))
       x_s = self.__dict__['_modules']["iter_%s" % str(i+1)](x_s) * torch.sigmoid(x_g_r + x_g_s) + x_s

    if self.downsample:
      x_s_1 = self.d_conv(self.d_ave(self.d_relu(self.d_bn(x_s))))
      x_s_2 = self.d_conv_1e(self.d_conv_3(self.d_relu_3(self.d_bn_3(self.d_conv_1(self.d_relu_1(self.d_bn_1(x_s)))))))
      x_s = x_s_1 + x_s_2

    return x_s

class GRCNNBackEnd(nn.Module):
  def __init__(self, iters, maps, SKconv, expansion, num_classes):
    """ Args:
      iters:iterations.
      num_classes: number of classes
    """
    super(GRCNNBackEnd, self).__init__()
    self.iters = iters
    self.maps = maps
    self.num_classes = num_classes
    self.expansion = expansion

 

    self.layer1 = GRCL(64, self.maps[0], True, self.iters[0], SKconv, self.expansion)
    self.layer2 = GRCL(self.maps[0] * self.expansion, self.maps[1], True, self.iters[1], SKconv, self.expansion)
    self.layer3 = GRCL(self.maps[1] * self.expansion, self.maps[2], True, self.iters[2], SKconv, self.expansion)
    self.layer4 = GRCL(self.maps[2] * self.expansion, self.maps[3], False, self.iters[3], SKconv, self.expansion)

    self.lastact = nn.Sequential(nn.BatchNorm2d(self.maps[3]*self.expansion), nn.ReLU(inplace=True))
    self.avgpool = nn.AvgPool2d(7)
    self.classifier = nn.Linear(self.maps[3] * self.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m.bias is not None:
           init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        init.zeros_(m.bias)

  def forward(self, x):

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.lastact(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

def grcnn55BackEnd(num_classes=1000):
  """
  Args:
    num_classes (uint): number of classes
  """
  model = GRCNNBackEnd([3, 3, 4, 3], [64, 128, 256, 512], SKconv=False, expansion=4, num_classes=num_classes)
  return model



###DEFINE YOUR CUSTOM MODEL HERE


preprocessing = functools.partial(load_preprocess_images, image_size=224)


vone_block, bottleneck = VOneNet()
model_back_end = grcnn55BackEnd()

model = nn.Sequential(OrderedDict([
                ('vone_block', vone_block),
                ('bottleneck', bottleneck),
                ('model', model_back_end),
            ]))
    
model = nn.Sequential(OrderedDict([('module',model)]))
    
dir_path = os.path.dirname(os.path.realpath(__file__))
state_dict = torch.load(dir_path + "/model_best.pth", map_location=device)['state_dict']
model.load_state_dict(state_dict)
model = model.to(device)








##### LIST YOUR CUSTOM LAYER HERE
#all_layers = [layer for layer, _ in model.named_modules()]
#print(all_layers)
#all_layers = all_layers[1:]

all_layers = ['module.bottleneck','module.model.layer2', 'module.model.layer2.conv_g_r',  'module.model.layer2.iter_1.8', 'module.model.layer2.iter_g_1.1','module.model.layer2', 'module.model.layer2.iter_2.8' ,'module.model.layer2.iter_g_2.1', 'module.model.layer2.iter_3.8', 
              'module.model.layer2.iter_g_3.1', 'module.model.layer3','module.model.layer3.conv_f', 'module.model.layer3.conv_g_r', 'module.model.layer3.iter_1.8', 'module.model.layer3.iter_g_1.1' ,
              'module.model.layer3.iter_2.8', 'module.model.layer3.iter_g_2.1', 'module.model.layer3.iter_3.8', 'module.model.layer3.iter_g_3.1', 'module.model.layer3.iter_4.8', 
              'module.model.layer3.iter_g_4.1', 'module.model.layer3.d_conv_1e', 'module.model.layer4', 'module.model.layer4.conv_g_r', 'module.model.layer4.iter_1.8', 'module.model.layer4.iter_g_1.1', 
              'module.model.layer4.iter_2.8', 'module.model.layer4.iter_g_2.1', 'module.model.layer4.iter_3.8', 'module.model.layer4.iter_g_3.1', 
              'module.model.lastact']

#'module.vone_block'
#, 'module.model.avgpool',  'module.model.classifier'
#print(all_layers)'''





# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier=model_identifier, model= model,
                                   preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier=model_identifier, activations_model=activations_model,
                        # specify layers to consider
                        layers=all_layers)





# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():

    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return [model_identifier]


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == model_identifier
    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    
    

    return wrapper



# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
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

    # quick check to make sure the model is the correct one:
    assert name == model_identifier

    # returns the layers you want to consider
    return  all_layers

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
