# encoding: utf8
"""
PyTorch implementation of the HMAX model of human vision. For more information
about HMAX, check:

    http://maxlab.neuro.georgetown.edu/hmax.html

The S and C units of the HMAX model can almost be mapped directly onto
TorchVision's Conv2d and MaxPool2d layers, where channels are used to store the
filters for different orientations. However, HMAX also implements multiple
scales, which doesn't map nicely onto the existing TorchVision functionality.
Therefore, each scale has its own Conv2d layer, which are executed in parallel.

Here is a schematic overview of the network architecture:

layers consisting of units with increasing scale
S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1 S1
 \ /   \ /   \ /   \ /   \ /   \ /   \ /   \ /
  C1    C1    C1    C1    C1    C1    C1    C1
   \     \     \    |     /     /     /     /
           ALL-TO-ALL CONNECTIVITY
   /     /     /    |     \     \     \     \
  S2    S2    S2    S2    S2    S2    S2    S2
   |     |     |     |     |     |     |     |
  C2    C2    C2    C2    C2    C2    C2    C2

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
Original repository: https://github.com/wmvanvliet/pytorch_hmax
(no license present when code was pulled). 

Edited by Jenelle Feather <jfeather@mit.edu> to fix euclidean distance and 
to include more stable operations for gradient calculations. Also included 
padding to match that used in the matlab version. 

References
----------

  .. [1] Riesenhuber, Maximilian, and Tomaso Poggio. “Hierarchical Models of
         Object Recognition in Cortex.” Nature Neuroscience 2, no. 11 (1999):
         1019–25.  https://doi.org/10.1038/14819.
  .. [2] Serre, T, M Kouh, C Cadieu, U Knoblich, Gabriel Kreiman, and T Poggio.
         “A Theory of Object Recognition: Computations and Circuits in the
         Feedforward Path of the Ventral Stream in Primate Visual Cortex.”
         Artificial Intelligence, no. December (2005): 1–130.
         https://doi.org/10.1.1.207.9279.
  .. [3] Serre, Thomas, Aude Oliva, and Tomaso Poggio. “A Feedforward
         Architecture Accounts for Rapid Categorization.” Proceedings of the
         National Academy of Sciences 104, no. 15 (April 10, 2007): 6424–29.
         https://doi.org/10.1073/pnas.0700622104.
  .. [4] Serre, Thomas, and Maximilian Riesenhuber. “Realistic Modeling of
         Simple and Complex Cell Tuning in the HMAXModel, and Implications for
         Invariant Object Recognition in Cortex.” CBCL Memo, no. 239 (2004).
  .. [5] Serre, Thomas, Lior Wolf, Stanley Bileschi, Maximilian Riesenhuber,
         and Tomaso Poggio. “Robust Object Recognition with Cortex-like
         Mechanisms.” IEEE Trans Pattern Anal Mach Intell 29, no. 3 (2007):
         411–26.  https://doi.org/10.1109/TPAMI.2007.56.
"""
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
import os

__all__ = ['hmax_standard_with_readout']

STABILITY_OFFSET = 1e-10

class ClippedPower(torch.autograd.Function):
    """
    Takes the power of a signal and clips its gradients to the 
    provided values in the backwards pass
    """
    @staticmethod
    def forward(ctx, x, clip_value, power):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        ctx.power = power
        return torch.pow(x, power)
 
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = ctx.power * torch.pow(x, ctx.power-1)
        return grad_output * torch.clamp(g, -ctx.clip_value, ctx.clip_value), None, None


def gabor_filter(size, wavelength, orientation):
    """Create a single gabor filter.

    Parameters
    ----------
    size : int
        The size of the filter, measured in pixels. The filter is square, hence
        only a single number (either width or height) needs to be specified.
    wavelength : float
        The wavelength of the grating in the filter, relative to the half the
        size of the filter. For example, a wavelength of 2 will generate a
        Gabor filter with a grating that contains exactly one wave. This
        determines the "tightness" of the filter.
    orientation : float
        The orientation of the grating in the filter, in degrees.

    Returns
    -------
    filt : ndarray, shape (size, size)
        The filter weights.
    """
    lambda_ = size * 2. / wavelength
    sigma = lambda_ * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    filt = np.exp(-(rotx**2 + gamma**2 * roty**2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lambda_)
    filt[np.sqrt(x**2 + y**2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt


class S1(nn.Module):
    """A layer of S1 units with different orientations but the same scale.

    The S1 units are at the bottom of the network. They are exposed to the raw
    pixel data of the image. Each S1 unit is a Gabor filter, which detects
    edges in a certain orientation. They are implemented as PyTorch Conv2d
    modules, where each channel is loaded with a Gabor filter in a specific
    orientation.

    Parameters
    ----------
    size : int
        The size of the filters, measured in pixels. The filters are square,
        hence only a single number (either width or height) needs to be
        specified.
    wavelength : float
        The wavelength of the grating in the filter, relative to the half the
        size of the filter. For example, a wavelength of 2 will generate a
        Gabor filter with a grating that contains exactly one wave. This
        determines the "tightness" of the filter.
    orientations : list of float
        The orientations of the Gabor filters, in degrees.
    """
    def __init__(self, size, wavelength, orientations=[90, -45, 0, 45],
                 include_borders=True):
        super().__init__()
        self.num_orientations = len(orientations)
        self.size = size
        # Added by jfeather to better match matlab implementation of padding 
        # (10/29/2021)
        self.include_borders = include_borders 

        # Use PyTorch's Conv2d as a base object. Each "channel" will be an
        # orientation. 'same' convolution is not standard in pytorch, so an 
        # approximation is used.
        self.gabor = nn.Conv2d(1, self.num_orientations, size,
                               padding=size // 2, bias=False)

        if not self.include_borders:
            self.remove_padding_amount = size // 2

        # Fill the Conv2d filter weights with Gabor kernels: one for each
        # orientation
        for channel, orientation in enumerate(orientations):
            self.gabor.weight.data[channel, 0] = torch.Tensor(
                gabor_filter(size, wavelength, orientation))

        # A convolution layer filled with ones. This is used to normalize the
        # result in the forward method.
        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # Since everything is pre-computed, no gradient is required
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, img):
        """Apply Gabor filters, take absolute value, and normalize."""
        s1_output = torch.abs(self.gabor(img))
        if not self.include_borders:
            s1_output[:,:,0:self.remove_padding_amount,:] = 0
            s1_output[:,:,:,0:self.remove_padding_amount] = 0
            s1_output[:,:,-self.remove_padding_amount:,:] = 0
            s1_output[:,:,:,-self.remove_padding_amount:] = 0
 
        # relu added by jfeather on 09/29/2021 to avoid nans (boundaries can cause values < 0)
        norm = torch.sqrt(torch.nn.functional.relu(self.uniform(img ** 2)) + STABILITY_OFFSET) 
        s1_output /= norm
        return s1_output


class C1(nn.Module):
    """A layer of C1 units with different orientations but the same scale.

    Each C1 unit pools over the S1 units that are assigned to it.

    Parameters
    ----------
    size : int
        Size of the MaxPool2d operation being performed by this C1 layer.
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.local_pool = nn.MaxPool2d(size, stride=size // 2,
                                       padding=size // 2)

    def forward(self, s1_outputs):
        """Max over scales, followed by a MaxPool2d operation."""
        s1_outputs = torch.cat([out.unsqueeze(0) for out in s1_outputs], 0)

        # Pool over all scales
        s1_output, _ = torch.max(s1_outputs, dim=0)

        # Pool over local (c1_space x c1_space) neighbourhood
        return self.local_pool(s1_output)


class S2(nn.Module):
    """A layer of S2 units with different orientations but the same scale.

    The activation of these units is computed by taking the distance between
    the output of the C layer below and a set of predefined patches. This
    distance is computed as:

      d = sqrt( (w - p)^2 )
        = sqrt( w^2 - 2pw + p^2 )

    Parameters
    ----------
    patches : ndarray, shape (n_patches, n_orientations, size, size)
        The precomputed patches to lead into the weights of this layer.
    activation : 'gaussian' | 'euclidean'
        Which activation function to use for the units. In the PNAS paper, a
        gaussian curve is used ('guassian', the default), whereas the MATLAB
        implementation of The Laboratory for Computational Cognitive
        Neuroscience uses the euclidean distance ('euclidean').
    sigma : float
        The sharpness of the tuning (sigma in eqn 1 of [1]_). Defaults to 1.

    References:
    -----------

    .. [1] Serre, Thomas, Aude Oliva, and Tomaso Poggio. “A Feedforward
           Architecture Accounts for Rapid Categorization.” Proceedings of the
           National Academy of Sciences 104, no. 15 (April 10, 2007): 6424–29.
           https://doi.org/10.1073/pnas.0700622104.
    """
    def __init__(self, patches, activation='gaussian', sigma=1):
        super().__init__()
        self.activation = activation
        self.sigma = sigma

        num_patches, num_orientations, size, _ = patches.shape

        # Main convolution layer
        self.conv = nn.Conv2d(in_channels=num_orientations,
                              out_channels=num_orientations * num_patches,
                              kernel_size=size,
                              padding=size // 2,
                              groups=num_orientations,
                              bias=False)
        self.conv.weight.data = torch.Tensor(
            patches.transpose(1, 0, 2, 3).reshape(1600, 1, size, size))

        # A convolution layer filled with ones. This is used for the distance
        # computation
        self.uniform = nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # This is also used for the distance computation
        self.patches_sum_sq = nn.Parameter(
            torch.Tensor((patches ** 2).sum(axis=(1, 2, 3))))

        self.num_patches = num_patches
        self.num_orientations = num_orientations
        self.size = size

        # No gradient required for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, c1_outputs):
        s2_outputs = []
        for c1_output in c1_outputs:
            conv_output = self.conv(c1_output)

            # Unstack the orientations
            conv_output_size = conv_output.shape[3]
            conv_output = conv_output.view(
                -1, self.num_orientations, self.num_patches, conv_output_size,
                conv_output_size)

            # Pool over orientations
            conv_output = conv_output.sum(dim=1)

            # Compute distance
            c1_sq = self.uniform(
                torch.sum(c1_output ** 2, dim=1, keepdim=True))
            dist = c1_sq - 2 * conv_output
            dist += self.patches_sum_sq[None, :, None, None]

            # Apply activation function
            if self.activation == 'gaussian':
                dist = torch.exp(- 1 / (2 * self.sigma ** 2) * dist)
            elif self.activation == 'euclidean':
                dist = torch.nn.functional.relu(dist)
                dist = torch.sqrt(dist + STABILITY_OFFSET)
                # negate the distance and then take the max value (matlab code takes the minimum)
                dist = -dist
            else:
                raise ValueError("activation parameter should be either "
                                 "'gaussian' or 'euclidean'.")

            s2_outputs.append(dist)
        return s2_outputs


class C2(nn.Module):
    """A layer of C2 units operating on a layer of S2 units."""
    def __init__(self, S2_activation='gaussian'):
        super().__init__()
        # If S2 activation is "euclidean" then we need to negate the output
        self.S2_activation = S2_activation
    def forward(self, s2_outputs):
        """Take the maximum value of the underlying S2 units."""
        maxs = [s2.max(dim=3)[0] for s2 in s2_outputs]
        maxs = [m.max(dim=2)[0] for m in maxs]
        maxs = torch.cat([m[:, None, :] for m in maxs], 1)
        if self.S2_activation=='euclidean':
            return -maxs.max(dim=1)[0]
        else:
            return maxs.max(dim=1)[0]


class HMAX(nn.Module):
    """The full HMAX model.

    Use the `get_all_layers` method to obtain the activations for all layers.

    If you are only interested in the final output (=C2 layer), use the model
    as any other PyTorch module:

        model = HMAX(universal_patch_set)
        output = model(img)

    Parameters
    ----------
    universal_patch_set : str
        Filename of the .mat file containing the universal patch set.
    s2_act : 'gaussian' | 'euclidean'
        The activation function for the S2 units. Defaults to 'gaussian'.

    Returns
    -------
    c2_output : list of Tensors, shape (batch_size, num_patches)
        For each scale, the output of the C2 units.
    """
    def __init__(self, universal_patch_set, s2_act='gaussian', 
                 linear_readout=False, num_classes=1000, 
                 include_S1_borders=True):
        super().__init__()
        self.linear_readout = linear_readout
        self.num_classes = num_classes

        # S1 layers, consisting of units with increasing size
        self.s1_units = nn.ModuleList([
            S1(size=7, wavelength=4, include_borders=include_S1_borders),
            S1(size=9, wavelength=3.95, include_borders=include_S1_borders),
            S1(size=11, wavelength=3.9, include_borders=include_S1_borders),
            S1(size=13, wavelength=3.85, include_borders=include_S1_borders),
            S1(size=15, wavelength=3.8, include_borders=include_S1_borders),
            S1(size=17, wavelength=3.75, include_borders=include_S1_borders),
            S1(size=19, wavelength=3.7, include_borders=include_S1_borders),
            S1(size=21, wavelength=3.65, include_borders=include_S1_borders),
            S1(size=23, wavelength=3.6, include_borders=include_S1_borders),
            S1(size=25, wavelength=3.55, include_borders=include_S1_borders),
            S1(size=27, wavelength=3.5, include_borders=include_S1_borders),
            S1(size=29, wavelength=3.45, include_borders=include_S1_borders),
            S1(size=31, wavelength=3.4, include_borders=include_S1_borders),
            S1(size=33, wavelength=3.35, include_borders=include_S1_borders),
            S1(size=35, wavelength=3.3, include_borders=include_S1_borders),
            S1(size=37, wavelength=3.25, include_borders=include_S1_borders),
        ])

        # Each C1 layer pools across two S1 layers
        self.c1_units = nn.ModuleList([
            C1(size=8),
            C1(size=10),
            C1(size=12),
            C1(size=14),
            C1(size=16),
            C1(size=18),
            C1(size=20),
            C1(size=22),
        ])

        # Read the universal patch set for the S2 layer
        m = loadmat(universal_patch_set)
        patches = [patch.reshape(shape[[2, 1, 0, 3]]).transpose(3, 0, 2, 1)
                   for patch, shape in zip(m['patches'][0], m['patchSizes'].T)]

        # One S2 layer for each patch scale, operating on all C1 layers
        self.s2_units = nn.ModuleList([S2(patches=scale_patches, activation=s2_act)
                         for scale_patches in patches])

        # One C2 layer operating on each scale
        self.c2_units = nn.ModuleList([C2(S2_activation=s2_act) for s2 in self.s2_units])

        if linear_readout:
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(3200, num_classes)

    def run_all_layers(self, img):
        """Compute the activation for each layer.

        Parameters
        ----------
        img : Tensor, shape (batch_size, 1, height, width)
            A batch of images to run through the model

        Returns
        -------
        s1_outputs : List of Tensors, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of S1 units.
        c1_outputs : List of Tensors, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of C1 units.
        s2_outputs : List of lists of Tensors, shape (batch_size, num_patches, height, width)
            For each C1 scale and each patch scale, the output of the layer of
            S2 units.
        c2_outputs : List of Tensors, shape (batch_size, num_patches)
            For each patch scale, the output of the layer of C2 units.
        """
        s1_outputs = [s1(img) for s1 in self.s1_units]

        # Each C1 layer pools across two S1 layers
        c1_outputs = []
        for c1, i in zip(self.c1_units, range(0, len(self.s1_units), 2)):
            c1_outputs.append(c1(s1_outputs[i:i+2]))

        s2_outputs = [s2(c1_outputs) for s2 in self.s2_units]
        c2_outputs = [c2(s2) for c2, s2 in zip(self.c2_units, s2_outputs)]

        return s1_outputs, c1_outputs, s2_outputs, c2_outputs

    def forward(self, img, with_latent=False, fake_relu=False, no_relu=False):
        """Run through everything and concatenate the output of the C2s."""
        all_outputs = {}
        all_outputs['input_after_preproc'] = img
        all_outputs['preproc_image'] = img
        layers_out = self.get_all_layers_tensors(img)
        if with_latent:
            batch_size = layers_out[0][0].shape[0]
            all_outputs['s1_out'] = torch.cat(layers_out[0], 1)
            all_outputs['c1_out_no_reshape'] = layers_out[1]
            all_outputs['c1_out'] = torch.cat([c[:, None, :].view(batch_size,-1) for c in layers_out[1]], 1)
            all_outputs['s2_out_no_reshape'] = layers_out[2]
            all_outputs['s2_out_reshape_to_list'] = [torch.cat([b.view(batch_size,-1) for b in a],1) for a in layers_out[2]]
            all_outputs['s2_out'] = torch.cat([b.view(batch_size,-1) for c in layers_out[2] for b in c], 1)
        c2_outputs = layers_out[-1]
        all_outputs['c2_out_no_reshape'] = layers_out[-1]
        c2_outputs = torch.cat(
            [c2_out[:, None, :] for c2_out in c2_outputs], 1)
        all_outputs['c2_out'] = c2_outputs

        if self.linear_readout:
            flattened_out = self.flatten(c2_outputs)
            linear_out = self.fc(flattened_out)
            all_outputs['final'] = linear_out
            if with_latent:
                return linear_out, c2_outputs, all_outputs
            return linear_out
        else:
            all_outputs['final'] = all_outputs['c2_out']
            if with_latent:
                return c2_outputs, None, all_outputs
            return c2_outputs

    def get_all_layers_tensors(self, img):
        """Get the activation for all layers as pytorch tensors.

        Parameters
        ----------
        img : Tensor, shape (batch_size, 1, height, width)
            A batch of images to run through the model

        Returns
        -------
        s1_outputs : List of tensors, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of S1 units.
        c1_outputs : List of tensors, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of C1 units.
        s2_outputs : List of lists of tensors, shape (batch_size, num_patches, height, width)
            For each C1 scale and each patch scale, the output of the layer of
            S2 units.
        c2_outputs : List of tensors, shape (batch_size, num_patches)
            For each patch scale, the output of the layer of C2 units.
        """
        s1_out, c1_out, s2_out, c2_out = self.run_all_layers(img)
        return s1_out, c1_out, s2_out, c2_out

    def get_all_layers(self, img):
        """Get the activation for all layers as NumPy arrays.

        Parameters
        ----------
        img : Tensor, shape (batch_size, 1, height, width)
            A batch of images to run through the model

        Returns
        -------
        s1_outputs : List of arrays, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of S1 units.
        c1_outputs : List of arrays, shape (batch_size, num_orientations, height, width)
            For each scale, the output of the layer of C1 units.
        s2_outputs : List of lists of arrays, shape (batch_size, num_patches, height, width)
            For each C1 scale and each patch scale, the output of the layer of
            S2 units.
        c2_outputs : List of arrays, shape (batch_size, num_patches)
            For each patch scale, the output of the layer of C2 units.
        """
        s1_out, c1_out, s2_out, c2_out = self.run_all_layers(img)
        return (
            [s1.cpu().detach().numpy() for s1 in s1_out],
            [c1.cpu().detach().numpy() for c1 in c1_out],
            [[s2_.cpu().detach().numpy() for s2_ in s2] for s2 in s2_out],
            [c2.cpu().detach().numpy() for c2 in c2_out],
        )

def hmax_standard_with_readout(**kwargs):
    """Constructs the hmax vision model with a linear readout that can be trained
    This way we can train a transfer head on top of the model and measure class accuracy 
    on ImageNet. 

    Args:
        pretrained (bool): 
    """
    if not os.path.isfile('hmax_universal_path_set.mat'):
        import urllib.request
        patch_set_url = 'https://github.com/wmvanvliet/pytorch_hmax/raw/master/universal_patch_set.mat'
        urllib.request.urlretrieve(patch_set_url, 'hmax_universal_path_set.mat')

    saved_param = 'hmax_universal_path_set.mat'

    model = HMAX(saved_param, linear_readout=True, num_classes=1000, 
                 s2_act='gaussian', include_S1_borders=False)
    for name, param in model.named_parameters():
        if name in ['fc.bias', 'fc.weight']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in model.named_parameters():
        print('%s: %s'%(name, param.requires_grad))
 
    return model

