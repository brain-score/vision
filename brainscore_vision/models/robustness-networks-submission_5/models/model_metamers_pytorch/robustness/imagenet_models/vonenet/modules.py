
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import gabor_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class SequentialWithAllOutput(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        if kwargs.get('with_latent', False):
            for i in range(l):
                if i == 0:
                    input, _, all_outputs = vs[i](input, *args, **kwargs)
                elif i == l-1:
                    input, pre_out, all_outputs_new = vs[i](input, *args, **kwargs)
                    all_outputs = self._append_all_outputs(all_outputs, all_outputs_new)
                else:
                    input, _, all_outputs_new = vs[i](input, *args, **kwargs)
                    all_outputs = self._append_all_outputs(all_outputs, all_outputs_new)
            return input, pre_out, all_outputs
        else:
            for i in range(l):
                input = vs[i](input)
            return input

    def _append_all_outputs(self, all_outputs, all_outputs_new):
        for key in all_outputs_new:
            if key in all_outputs:
                raise ValueError('Specifying two key names for parts of ' \
                                 'all_outputs that are the same')
            all_outputs[key] = all_outputs_new[key]
        return all_outputs

class BottleneckVOne(nn.Conv2d):
    """Wrap add additional arguments for the forward pass of the Conv2d layer
    used as a bottleneck for the VOneBlock"""
    def forward(self, x, with_latent=False, fake_relu=False, no_relu=None):
        if with_latent:
            all_outputs = {}
            all_outputs['input_bottleneck'] = x
        x = self._conv_forward(x, self.weight)
        if with_latent:
            all_outputs['bottleneck'] = x
        if with_latent:
            return x, None, all_outputs
        return x

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
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224,
                 num_stochastic_copies=None):
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

        self.num_stochastic_copies = num_stochastic_copies

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=False)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = Identity()
        self.output = nn.ReLU(inplace=False)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=None):
        # TODO: fake_relu for output? 
        if with_latent:
            all_outputs = {}
            all_outputs['input_after_preproc'] = x

        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        if with_latent:
            all_outputs['gabors_f'] = x

        # Noise [Batch, out_channels, H/stride, W/stride]
        if self.num_stochastic_copies is not None:
            x = self.noise_f(x.repeat(self.num_stochastic_copies, 1, 1, 1))
        else:
            x = self.noise_f(x)
        if with_latent:
            all_outputs['noise_f'] = x

        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        if with_latent and fake_relu:
            all_outputs['v1_output_fake_relu'] = FakeReLU.apply(x)
        x = self.output(x)
        if with_latent:
            all_outputs['v1_output'] = x

        if with_latent:
            return x, None, all_outputs
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(0.00001+ s_q0[:, self.simple_channels:, :, :] ** 2 +
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
        elif self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=self.noise_level).rsample()
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
#         noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride), int(self.input_size/self.stride))
        # use broadcasting -- everything in the batch has the same noise. 
        if self.num_stochastic_copies is None:
            noise_mean = torch.zeros(1, self.out_channels, int(self.input_size/self.stride), int(self.input_size/self.stride))
        else:
            noise_mean = torch.zeros(self.num_stochastic_copies, self.out_channels, int(self.input_size/self.stride), int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode == 'gaussian': 
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=self.noise_level).rsample().to(device)
        elif self.noise_mode == 'neuronal':
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)
        print(self.fixed_noise[:,0,0,0])

    def unfix_noise(self):
        self.fixed_noise = None
