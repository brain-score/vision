import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .utils import gaussian_kernel, circular_kernel, gabor_kernel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-4

class Identity(nn.Module):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase, color):
        for i in range(self.out_channels):
            self.weight[i, int(color[i])] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                            theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(
        self, sf, theta, sigx, sigy, phase, color, in_channels=3,
        k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
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

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None
        
        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase, color=self.color)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2, color=self.color)

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


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False, groups=groups)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.conv(x)

class DoGFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, across_channels=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.across_channels = across_channels
        if across_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False, padding_mode='reflect')
        else:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', bias=False, padding_mode='reflect', groups=in_channels)
    
    def initialize(self, r_c: torch.Tensor,  r_s: torch.Tensor,  opponency: torch.Tensor) :
        assert r_c.size()[0] == r_s.size()[0] == self.out_channels
        assert opponency.shape[0] == self.out_channels
        if self.across_channels:
            for i in range(self.out_channels):
                center = gaussian_kernel(sigma=r_c[i]/np.sqrt(2), size=self.kernel_size, norm=False)
                surround = gaussian_kernel(sigma=r_s[i]/np.sqrt(2), size=self.kernel_size, norm=False)
                kernels = tuple((c * center + s * surround) for c, s in zip(*opponency[i]))
                self.conv.weight.data[i] = torch.stack(kernels, dim=0)
                self.conv.weight.data[i] /= torch.abs(self.conv.weight.data[i].sum())
        else:
            center = [
                gaussian_kernel(sigma=r_c[i]/np.sqrt(2), size=self.kernel_size, norm=False)
                for i in range(self.in_channels)
                ]
            surround = [
                gaussian_kernel(sigma=r_s[i]/np.sqrt(2), size=self.kernel_size, norm=False)
                for i in range(self.in_channels)
                ]
            kernels = tuple(
                (c * center[i] + s * surround[i])
                for i, (c, s) in enumerate(zip(*opponency.sum(dim=2).T))
                )
            self.conv.weight.data = torch.stack(kernels, dim=0).unsqueeze(1)
            self.conv.weight.data /= torch.abs(self.conv.weight.data.sum(dim=(2, 3), keepdim=True))
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class LightAdaptation(nn.Module):
    def __init__(self, in_channels, kernel_size, radius, gaussian=True):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        if radius == np.inf:
            self.mean = lambda x: torch.mean(x, axis=(1, 2, 3), dtype=torch.float).view(x.size()[0], 1, 1, 1)
            return
        filter = gaussian_kernel(radius/np.sqrt(2), size=kernel_size, norm=True) if gaussian else circular_kernel(kernel_size, radius)
        self.conv = nn.Conv2d(1, 1, kernel_size, padding='same', bias=False, padding_mode='reflect')
        self.conv.weight.data = filter[None, None, ...]
        self.conv.weight.requires_grad = False
        self.mean = lambda x: self.conv(torch.mean(x, axis=1, keepdim=True))

    def forward(self, x):
        return (x / (self.mean(x) + EPSILON)) - 1


class ContrastNormalization(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int, radius:int, c50:float):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.c50 = c50
        self.radius = radius

        self.conv = nn.Conv2d(in_channels, 1, kernel_size, padding='same', bias=False, padding_mode='reflect')
        filter = gaussian_kernel(sigma=radius/np.sqrt(2), size=self.kernel_size)
        self.conv.weight.data = filter.expand(self.in_channels, -1, -1).unsqueeze(0) / (torch.sum(filter) * in_channels)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x / (torch.abs(self.conv(y ** 2)) ** .5 + self.c50)


class MixedDoG(nn.Module):
    def __init__(self, dog_p_cells: nn.Module, dog_m_cells: nn.Module):
        super().__init__()
        self.dog_p_cells = dog_p_cells
        self.dog_m_cells = dog_m_cells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hstack((self.dog_p_cells(x), self.dog_m_cells(torch.mean(x, dim=1, keepdim=True))))


class MixedContrastNormalization(nn.Module):
    def __init__(self, contrast_norm_p: nn.Module, contrast_norm_m: nn.Module):
        super().__init__()
        self.contrast_norm_p = contrast_norm_p
        self.contrast_norm_m = contrast_norm_m

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.hstack((self.contrast_norm_p(x[:, :3], y[:, :3]), self.contrast_norm_m(x[:, 3:], y[:, 3:])))

class RetinaBlock(nn.Module):
    def __init__(
        self, in_channels, m_channels, p_channels,
        dog_across_channels,
        rc_p_cell, rs_p_cell, opponency_p_cell, kernel_p_cell,  # P-Cell Difference of Gaussians
        rc_m_cell, rs_m_cell, opponency_m_cell, kernel_m_cell,  # M-Cell Difference of Gaussians
        kernel_la, radius_la,  # Light Adaptation Layer
        kernel_cn, radius_cn, c50,  # Contrast Normalization Layer
        linear_p_cells=False,
        light_adapt=True, dog=True, contrast_norm=False
        ):
        super().__init__()
        self.in_channels = in_channels
        self.p_channels = p_channels
        self.m_channels = m_channels

        # Layers
        self.light_adapt = Identity()
        self.dog_p_cells = Identity()
        self.dog_m_cells = Identity()
        self.dog = Identity()
        self.contrast_norm = Identity()

        # DoG Filter Bank
        if dog:
            self.rc_p_cell = rc_p_cell
            self.rs_p_cell = rs_p_cell
            self.opponency_p_cell = opponency_p_cell
            self.kernel_p_cell = kernel_p_cell
            self.dog_p_cells = DoGFB(self.in_channels, self.p_channels, self.kernel_p_cell, across_channels=dog_across_channels)
            self.dog_p_cells.initialize(r_c=self.rc_p_cell, r_s=self.rs_p_cell, opponency=self.opponency_p_cell)
            if m_channels > 0:
                self.rc_m_cell = rc_m_cell
                self.rs_m_cell = rs_m_cell
                self.opponency_m_cell = opponency_m_cell
                self.kernel_m_cell = kernel_m_cell
                self.dog_m_cells = DoGFB(1, self.m_channels, self.kernel_m_cell, across_channels=False)
                self.dog_m_cells.initialize(r_c=self.rc_m_cell, r_s=self.rs_m_cell, opponency=self.opponency_m_cell)
                self.dog = MixedDoG(self.dog_p_cells, self.dog_m_cells)
            else:
                self.dog = self.dog_p_cells

        # Light Adaptation
        if light_adapt:
            self.kernel_la = kernel_la
            self.light_adapt = LightAdaptation(self.in_channels, self.kernel_la, radius_la)

        # Contrast Normalization
        if contrast_norm:
            self.kernel_cn = kernel_cn
            self.radius_cn = radius_cn
            self.c50 = c50
            if not m_channels:
                    self.contrast_norm = ContrastNormalization(p_channels, self.kernel_p_cell, radius=self.rs_p_cell[0].item(), c50=self.c50)
            else:
                if linear_p_cells:
                    self.contrast_norm = MixedContrastNormalization(
                        Identity(),
                        ContrastNormalization(m_channels, self.kernel_m_cell, radius=self.rs_m_cell[0].item(), c50=self.c50)
                        )
                else:
                    self.contrast_norm = MixedContrastNormalization(
                        ContrastNormalization(p_channels, self.kernel_p_cell, radius=self.rs_p_cell[0].item(), c50=self.c50),
                        ContrastNormalization(m_channels, self.kernel_m_cell, radius=self.rs_m_cell[0].item(), c50=self.c50)
                        )
        
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.light_adapt(x)
        x = self.dog(x)
        x = self.contrast_norm(x, x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
