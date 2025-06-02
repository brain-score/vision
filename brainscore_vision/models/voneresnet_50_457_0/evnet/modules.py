import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from .utils import gaussian_kernel, circular_kernel, gabor_kernel

EPSILON = 1e-5

class Identity(nn.Module):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x


class MultiKernelConv2D(nn.Module):
    def __init__(self, in_channels_idx, out_channels_idx, kernel_sizes, paddings, bias, padding_modes, groups=None):
        super().__init__()
        if not groups: groups = [1]*len(in_channels_idx)
        #print(len(in_channels_idx), len(out_channels_idx), len(kernel_sizes), len(paddings), len(bias), len(padding_modes), len(groups))
        assert len(in_channels_idx)==len(out_channels_idx)==len(kernel_sizes)==len(paddings)==len(bias)==len(padding_modes)==len(groups)
        self.in_channels_idx = in_channels_idx
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=len(in_ch), out_channels=out_ch, kernel_size=k, padding=p, groups=g, bias=b, padding_mode=m)
            for in_ch, out_ch, k, p, g, b, m in zip(in_channels_idx, out_channels_idx, kernel_sizes, paddings, groups, bias, padding_modes)
        ])

    def forward(self, x):
        outputs = [conv(x[:, self.in_channels_idx[c]]) for c, conv in enumerate(self.convs)]
        return torch.cat(outputs, dim=1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase, color, complex_color2=False):
        # random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        if complex_color2:
            for i in range(self.out_channels//2):
                self.weight[i, int(color[i])] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
            for i in range(self.out_channels//2, self.out_channels):
                self.weight[i, 0] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
                self.weight[i, 1] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
                self.weight[i] /= 2
        else:
            for i in range(self.out_channels):
                self.weight[i, int(color[i])] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase, color, in_channels=3,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=0, fano_factor=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224
                 ):
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
        self.color = color
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level, fano_factor)
        self.fixed_noise = None
        
        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase, color=self.color)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2, color=self.color)

        self.simple = nn.ReLU(inplace=False)
        self.complex = Identity()
        self.gabors = Identity()
        #self.noise = nn.ReLU(inplace=True)
        self.noise = Identity()
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
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(self.fano_factor * F.relu(x.clone()) + EPSILON)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(self.fano_factor * F.relu(x.clone()) + EPSILON)
            #x -= self.noise_level
            #x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1, fano_factor=1):
        if not noise_mode: print(f'[VOneBlock] Using noise_mode={noise_mode}.')
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level
        self.fano_factor = fano_factor

    def fix_noise(self, batch_size=128, seed=42):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(torch.cuda.current_device())

    def unfix_noise(self):
        self.fixed_noise = None


class DoG(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, kernel_size, r_c,  r_s, opponency):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        assert r_c.shape[0] == r_s.shape[0] == num_conv
        assert opponency.shape[0] == p_channels+m_channels
        self.in_channels = in_channels
        self.p_channels = p_channels
        self.m_channels = m_channels
        self.kernel_size = kernel_size
        self.r_c = r_c
        self.r_s = r_s
        self.opponency = opponency

        self.multi_conv = MultiKernelConv2D(
            [torch.arange(in_channels)]*bool(p_channels)+\
            [torch.arange(in_channels, in_channels*2)]*bool(m_channels),
            [p_channels]*bool(p_channels)+[m_channels]*bool(m_channels), kernel_size,
            ['same']*num_conv, [False]*num_conv, ['reflect']*num_conv
            )
        for i in range(p_channels+m_channels):
            conv_i = i // p_channels
            filter_i = i % p_channels
            conv = self.multi_conv.convs[conv_i]
            center = gaussian_kernel(sigma=r_c[conv_i]/np.sqrt(2), size=kernel_size[conv_i], norm=False)
            surround = gaussian_kernel(sigma=r_s[conv_i]/np.sqrt(2), size=kernel_size[conv_i], norm=False)
            kernels = tuple((c * center + s * surround) for c, s in zip(*opponency[i]))
            conv.weight.data[filter_i] = torch.stack(kernels, dim=0)
            conv.weight.data[filter_i] /= torch.abs(conv.weight.data[filter_i].sum())
            conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multi_conv(x)


class LightAdaptation(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, kernel_sizes, radii, n, gaussian=True):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        self.in_channels = in_channels
        self.kernel_size = kernel_sizes
        self.radii = radii
        self.n = nn.Parameter(torch.from_numpy(n).view(1, num_conv, 1, 1), requires_grad=False)
        if np.isinf(radii).all().item():
            self.mean = lambda x: torch.mean(x, axis=(1, 2, 3), keepdim=True, dtype=torch.float).expand(-1, num_conv, -1, -1)
            return
        
        self.mean = MultiKernelConv2D(
            [[0]]*num_conv, [1]*num_conv, kernel_sizes, ['same']*num_conv, [False]*num_conv, ['reflect']*num_conv
            )
        for i, conv in enumerate(self.mean.convs):
            filter = gaussian_kernel(radii[i]/np.sqrt(2), size=kernel_sizes[i], norm=True)\
                if gaussian else circular_kernel(kernel_sizes[i], radii[i])
            conv.weight.data = filter[None, None, ...]
            conv.weight.requires_grad = False

    def forward(self, x):
        num = x.unsqueeze(1) ** self.n.unsqueeze(2)
        denom = (self.mean(x) ** self.n).unsqueeze(2) + num + EPSILON
        return  (num / denom).reshape(x.shape[0], -1, *x.shape[2:]) - .5


class ContrastNormalization(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, kernel_sizes, radii, c50, n):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        self.in_channels = in_channels
        self.kernel_size = kernel_sizes
        self.radius = radii
        select_idxs = torch.tensor([0]*p_channels+[1]*m_channels)
        self.c50 = torch.index_select(torch.from_numpy(c50), 0, select_idxs)
        self.c50 = nn.Parameter(self.c50.view(1, p_channels+m_channels, 1, 1), requires_grad=False)
        self.n = torch.index_select(torch.from_numpy(n), 0, select_idxs)
        self.n = nn.Parameter(self.n.view(1, p_channels+m_channels, 1, 1), requires_grad=False)

        idxs = [p_channels]*bool(p_channels)+[m_channels]*bool(m_channels)
        self.multi_conv = MultiKernelConv2D(
            torch.arange(p_channels+m_channels).split(idxs), idxs,
            kernel_sizes, ['same']*num_conv, [False]*num_conv, ['reflect']*num_conv, idxs
            )
        for conv_i, (num_channels, conv) in enumerate(zip((p_channels, m_channels), self.multi_conv.convs)):
            filter = gaussian_kernel(sigma=radii[conv_i]/np.sqrt(2), size=kernel_sizes[conv_i])
            conv.weight.data = filter.expand(num_channels, -1, -1).unsqueeze(1) / torch.sum(filter)
            conv.weight.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        return x / (self.multi_conv(x ** 2) ** .5 + self.c50)**self.n


class RetinaBlock(nn.Module):
    def __init__(
        self, in_channels, p_channels, m_channels, 
        rc_dog, rs_dog, opponency_dog, kernel_dog,
        kernel_la, radius_la, exp_la, kernel_cn, radius_cn, c50_cn, exp_cn,
        k_exc, fano_factor=.4, noise_mode=None,
        with_light_adapt=True, with_dog=True, with_contrast_norm=True, with_relu=True
        ):
        super().__init__()
        self.in_channels = in_channels
        self.p_channels = p_channels
        self.m_channels = m_channels
        self.noise_mode = noise_mode
        self.fano_factor = fano_factor
        self.fixed_noise = None
        self.k_exc = torch.tensor(k_exc, dtype=torch.float32).view(1, p_channels+m_channels, 1, 1)
        self.k_exc = nn.Parameter(self.k_exc, requires_grad=False)

        self.light_adapt = Identity()
        self.dog = Identity()
        self.contrast_norm = Identity()
        self.nonlinearity = Identity()

        if with_light_adapt:
            self.light_adapt = LightAdaptation(in_channels, p_channels, m_channels, kernel_la, radius_la, exp_la)
        if with_dog:
            self.dog = DoG(in_channels, p_channels, m_channels, kernel_dog, rc_dog, rs_dog, opponency_dog)
        if with_contrast_norm:
            self.contrast_norm = ContrastNormalization(in_channels, p_channels, m_channels, kernel_cn, radius_cn, c50_cn, exp_cn)
        if with_relu:
            self.nonlinearity = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.light_adapt(x)
        x = self.dog(x)
        x = self.contrast_norm(x)
        x = self.nonlinearity(x)
        x *= self.k_exc
        x = self.noise_f(x)
        return x

    def noise_f(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_mode:
            r = self.fixed_noise if self.fixed_noise is not None else\
                torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample()
            if self.noise_mode == 'additive':
                    x += r * torch.sqrt(self.fano_factor * torch.tensor(self.noise_level, dtype=x.dtype, device=x.device))
            elif self.noise_mode == 'multiplicative':
                    x += r * torch.sqrt(self.fano_factor * ((torch.abs(x.clone()) + EPSILON)))
        return x

    def fix_noise(self, image_size=224, batch_size=128, seed=42):
        noise_mean = torch.zeros(
            batch_size, self.p_channels+self.m_channels, image_size, image_size
            )
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(torch.cuda.current_device())
    
    def unfix_noise(self):
        self.fixed_noise = None
