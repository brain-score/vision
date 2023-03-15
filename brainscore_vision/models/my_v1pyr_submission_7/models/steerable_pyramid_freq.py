import warnings
from collections import OrderedDict
import numpy as np
from .v1_utils import interpolate1d, raised_cosine
import torch
import torch.fft as fft
import torch.nn as nn
from scipy.special import factorial


complex_types = [torch.complex64, torch.cdouble, torch.complex32, torch.cfloat]

class Steerable_Pyramid_Freq(nn.Module):
    r"""Steerable frequency pyramid in Torch

    Construct a steerable pyramid on matrix two dimensional signals, in the
    Fourier domain. Boundary-handling is circular. Reconstruction is exact
    (within floating point errors). However, if the image has an odd-shape,
    the reconstruction will not be exact due to boundary-handling issues
    that have not been resolved.

    The squared radial functions tile the Fourier plane with a raised-cosine
    falloff. Angular functions are cos(theta-k*pi/order+1)^(order).

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.
    For further information see the project webpage_

    Parameters
    ----------
    image_shape : `list or tuple`
        shape of input image
    height : 'auto' or `int`
        The height of the pyramid. If 'auto', will automatically determine
        based on the size of `image`.
    order : `int`.
        The Gaussian derivative order used for the steerable filters. Default value is 3.
        Note that to achieve steerability the minimum number of orientation is `order` + 1,
        and is used here. To get more orientations at the same order, use the method `steer_coeffs`
    twidth : `int`
        The width of the transition region of the radial lowpass function, in octaves
    is_complex : `bool`
        Whether the pyramid coefficients should be complex or not. If True, the real and imaginary
        parts correspond to a pair of even and odd symmetric filters. If False, the coefficients
        only include the real part / even symmetric filter.
    downsample: `bool`
        Whether to downsample each scale in the pyramid or keep the output pyramid coefficients
        in fixed bands of size imshapeximshape. When downsample is False, the forward method returns a tensor.
    tight_frame: `bool` default: False
        Whether the pyramid obeys the generalized parseval theorem or not (i.e. is a tight frame).
        If True, the energy of the pyr_coeffs = energy of the image. If not this is not true.
        In order to match the matlabPyrTools or pyrtools pyramids, this must be set to False

    Attributes
    ----------
    image_shape : `list or tuple`
        shape of input image
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
    fft_norm : `str`
        The way the ffts are normalized, see pytorch documentation for more details.
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.
    .. _webpage: https://www.cns.nyu.edu/~eero/steerpyr/

    """

    def __init__(self, image_shape, height='auto', order=3, twidth=1, is_complex=False,
                  downsample=True,  tight_frame=False):

        super().__init__()

        self.pyr_size = OrderedDict()
        self.order = order
        self.image_shape = image_shape

        if (self.image_shape[0] % 2 != 0) or (self.image_shape[1] % 2 != 0):
            warnings.warn(
                "Reconstruction will not be perfect with odd-sized images")

        self.is_complex = is_complex
        self.downsample = downsample
        self.tight_frame = tight_frame
        if self.tight_frame:
            self.fft_norm = "ortho"
        else:
            self.fft_norm = "backward"
        # cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * \
            np.array(range(-(2*self.lutsize + 1),
                           (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi

        max_ht = np.floor(np.log2(min(self.image_shape[0], self.image_shape[1])))-2
        if height == 'auto':
            self.num_scales = int(max_ht)
        elif height > max_ht:
            raise Exception(
                "Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

        if self.order > 15 or self.order <= 0:
            warnings.warn(
                "order must be an integer in the range [1,15]. Truncating.")
            self.order = min(max(self.order, 1), 15)
        self.num_orientations = int(self.order + 1)

        if twidth <= 0:
            warnings.warn("twidth must be positive. Setting to 1.")
            twidth = 1
        twidth = int(twidth)

        dims = np.array(self.image_shape)

        # make a grid for the raised cosine interpolation
        ctr = np.ceil((np.array(dims)+0.5)/2).astype(int)

        (xramp, yramp) = np.meshgrid(np.linspace(-1, 1, dims[1]+1)[:-1],
                                     np.linspace(-1, 1, dims[0]+1)[:-1])

        self.angle = np.arctan2(yramp, xramp)
        log_rad = np.sqrt(xramp**2 + yramp**2)
        log_rad[ctr[0]-1, ctr[1]-1] = log_rad[ctr[0]-1, ctr[1]-2]
        self.log_rad = np.log2(log_rad)

        # radial transition function (a raised cosine in log-frequency):
        self.Xrcos, Yrcos = raised_cosine(twidth, (-twidth/2.0), np.array([0, 1]))
        self.Yrcos = np.sqrt(Yrcos)

        self.YIrcos = np.sqrt(1.0 - self.Yrcos**2)

        # create low and high masks
        lo0mask = interpolate1d(self.log_rad, self.YIrcos, self.Xrcos)
        hi0mask = interpolate1d(self.log_rad, self.Yrcos, self.Xrcos)
        self.lo0mask = torch.tensor(lo0mask).unsqueeze(0)
        self.hi0mask = torch.tensor(hi0mask).unsqueeze(0)

        # pre-generate the angle, hi and lo masks, as well as the
        # indices used for down-sampling
        self._anglemasks = []
        self._anglemasks_recon = []
        self._himasks = []
        self._lomasks = []
        self._loindices = []

        # need a mock image to down-sample so that we correctly
        # construct the differently-sized masks
        mock_image = np.random.rand(*self.image_shape)
        imdft = np.fft.fftshift(np.fft.fft2(mock_image))
        lodft = imdft * lo0mask

        # this list, used by coarse-to-fine optimization, gives all the
        # scales (including residuals) from coarse to fine
        self.scales = (['residual_lowpass'] + list(range(self.num_scales))[::-1] +
                       ['residual_highpass'])

        # we create these copies because they will be modified in the
        # following loops
        Xrcos = self.Xrcos.copy()
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        for i in range(self.num_scales):
            Xrcos -= np.log2(2)
            const = ((2 ** (2*self.order)) * (factorial(self.order, exact=True)**2) /
                     float(self.num_orientations * factorial(2*self.order, exact=True)))

            if self.is_complex:
                Ycosn_forward = (2.0 * np.sqrt(const) * (np.cos(self.Xcosn) ** self.order) *
                                 (np.abs(self.alpha) < np.pi/2.0).astype(int))
                Ycosn_recon = np.sqrt(const) * (np.cos(self.Xcosn))**self.order

            else:
                Ycosn_forward = np.sqrt(
                    const) * (np.cos(self.Xcosn))**self.order
                Ycosn_recon = Ycosn_forward

            himask = interpolate1d(log_rad, self.Yrcos, Xrcos)
            self._himasks.append(torch.tensor(himask).unsqueeze(0))

            anglemasks = []
            anglemasks_recon = []
            for b in range(self.num_orientations):
                anglemask = interpolate1d(angle, Ycosn_forward,
                                    self.Xcosn + np.pi*b/self.num_orientations)
                anglemask_recon = interpolate1d(
                    angle, Ycosn_recon, self.Xcosn + np.pi*b/self.num_orientations)
                anglemasks.append(torch.tensor(anglemask).unsqueeze(0))
                anglemasks_recon.append(torch.tensor(anglemask_recon).unsqueeze(0))

            self._anglemasks.append(anglemasks)
            self._anglemasks_recon.append(anglemasks_recon)
            if not self.downsample:
                lomask = interpolate1d(log_rad, self.YIrcos, Xrcos)
                self._lomasks.append(torch.tensor(lomask).unsqueeze(0))
                self._loindices.append([np.array([0, 0]), dims])
                lodft = lodft * lomask

            else:
                # subsample lowpass
                dims = np.array([lodft.shape[0], lodft.shape[1]])
                ctr = np.ceil((dims+0.5)/2).astype(int)
                lodims = np.ceil((dims-0.5)/2).astype(int)
                loctr = np.ceil((lodims+0.5)/2).astype(int)
                lostart = ctr - loctr
                loend = lostart + lodims
                self._loindices.append([lostart, loend])

                # subsample indices
                log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
                angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

                lomask = interpolate1d(log_rad, self.YIrcos, Xrcos)
                self._lomasks.append(torch.tensor(lomask).unsqueeze(0))
                # subsampling
                lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
                # convolution in spatial domain
                lodft = lodft * lomask

        # reasonable default dtype
        self = self.to(torch.float32)

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.
        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self
        """
        self.lo0mask = self.lo0mask.to(*args, **kwargs)
        self.hi0mask = self.hi0mask.to(*args, **kwargs)
        self._himasks = [m.to(*args, **kwargs) for m in self._himasks]
        self._lomasks = [m.to(*args, **kwargs) for m in self._lomasks]
        angles = []
        angles_recon = []
        for a, ar in zip(self._anglemasks, self._anglemasks_recon):
            angles.append([m.to(*args, **kwargs) for m in a])
            angles_recon.append([m.to(*args, **kwargs) for m in ar])
        self._anglemasks = angles
        self._anglemasks_recon = angles_recon
        return self

    def forward(self, x, scales=[]):
        r"""Generate the steerable pyramid coefficients for an image

        Parameters
        ----------
        x : torch.Tensor
            A tensor containing the image to analyze. We want to operate
            on this in the pytorch-y way, so we want it to be 4d (batch,
            channel, height, width).
        scales : list, optional
            Which scales to include in the returned representation. If
            an empty list (the default), we include all
            scales. Otherwise, can contain subset of values present in
            this model's ``scales`` attribute (ints from 0 up to
            ``self.num_scales-1`` and the strs 'residual_highpass' and
            'residual_lowpass'. Can contain a single value or multiple
            values. If it's an int, we include all orientations from
            that scale. Order within the list does not matter.

        Returns
        -------
        representation: torch.Tensor or OrderedDict
            if the not downsampled version is used, representation is returned
            as a torch tensor with each band as a channel in BxCxHxW. The order
            of the channels is the same order as the keys in the pyr_coeffs dictonary.
            If the pyramid is complex, the channels are ordered such that for each band,
            the real channel comes first, followed by the imaginary channel.

            If downsample is true, representation is an OrderedDict of the coefficients.

        """
        pyr_coeffs = OrderedDict()
        if not isinstance(scales, list):
            raise Exception("scales must be a list!")
        if not scales:
            scales = self.scales
        scale_ints = [s for s in scales if isinstance(s, int)]
        if len(scale_ints) != 0:
            assert (max(scale_ints) < self.num_scales) and (
                min(scale_ints) >= 0), "Scales must be within 0 and num_scales-1"
        angle = self.angle.copy()
        log_rad = self.log_rad.copy()
        lo0mask = self.lo0mask.clone()
        hi0mask = self.hi0mask.clone()

        # x is a torch tensor batch of images of size [N,C,W,H]
        assert len(x.shape) == 4, "Input must be batch of images of shape BxCxHxW"
        
        imdft = fft.fft2(x, dim=(-2,-1), norm = self.fft_norm)
        imdft = fft.fftshift(imdft)
        
        if 'residual_highpass' in scales:
            # high-pass
            hi0dft = imdft * hi0mask
            hi0 = fft.ifftshift(hi0dft)
            hi0 = fft.ifft2(hi0, dim=(-2,-1), norm=self.fft_norm)
            pyr_coeffs['residual_highpass'] = hi0.real
            self.pyr_size['residual_highpass'] = tuple(hi0.real.shape[-2:])

        #input to the next scale is the low-pass filtered component
        lodft = imdft * lo0mask

        for i in range(self.num_scales):

            if i in scales:
                #high-pass mask is selected based on the current scale
                himask = self._himasks[i]
                #compute filter output at each orientation
                for b in range(self.num_orientations):
                    
                    # band pass filtering is done in the fourier space as multiplying by the fft of a gaussian derivative.
                    # The oriented dft is computed as a product of the fft of the low-passed component,
                    # the precomputed anglemask (specifies orientation), and the precomputed hipass mask (creating a bandpass filter)
                    # the complex_const variable comes from the Fourier transform of a gaussian derivative.
                    # Based on the order of the gaussian, this constant changes.
                    
                  
                    anglemask = self._anglemasks[i][b]
                    complex_const = np.power(complex(0, -1), self.order)
                    banddft = complex_const * lodft * anglemask * himask
                    # fft output is then shifted to center frequencies
                    band = fft.ifftshift(banddft)
                    # ifft is applied to recover the filtered representation in spatial domain
                    band = fft.ifft2(band, dim=(-2,-1), norm=self.fft_norm)
                    
                    #for real pyramid, take the real component of the complex band
                    if not self.is_complex:
                        pyr_coeffs[(i, b)] = band.real
                    else:
                        
                        # Because the input signal is real, to maintain a tight frame 
                        # if the complex pyramid is used, magnitudes need to be divided by sqrt(2) 
                        # because energy is doubled.
                
                        if self.tight_frame:
                            band = band/np.sqrt(2)
                        pyr_coeffs[(i, b)] = band
                    self.pyr_size[(i, b)] = tuple(band.shape[-2:])

            if not self.downsample:
                # no subsampling of angle and rad
                # just use lo0mask
                lomask = self._lomasks[i]
                lodft = lodft * lomask
                
                # because we don't subsample here, if we are not using orthonormalization that
                # we need to manually account for the subsampling, so that energy in each band remains the same
                # the energy is cut by factor of 4 so we need to scale magnitudes by factor of 2
                
                if self.fft_norm != "ortho":
                    lodft = 2*lodft
            else:
                # subsample indices
                lostart, loend = self._loindices[i]

                log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
                angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]

                # subsampling of the dft for next scale
                lodft = lodft[:, :, lostart[0]:loend[0], lostart[1]:loend[1]]
                # low-pass filter mask is selected
                lomask = self._lomasks[i]
                # again multiply dft by subsampled mask (convolution in spatial domain)

                lodft = lodft * lomask

        if 'residual_lowpass' in scales:
            # compute residual lowpass when height <=1
            lo0 = fft.ifftshift(lodft)
            lo0 = fft.ifft2(lo0, dim=(-2,-1), norm=self.fft_norm)
            pyr_coeffs['residual_lowpass'] = lo0.real
            self.pyr_size['residual_lowpass'] = tuple(lo0.real.shape[-2:])

        return pyr_coeffs
    
    @staticmethod
    def convert_pyr_to_tensor(pyr_coeffs, split_complex=False):
        r"""
        Function that takes a torch pyramid (without downsampling) dictonary
        and converts the output into a single tensor of BxCxHxW for use in an
        nn module downstream. In the multichannel case, all bands for each channel will be
        stacked together (i.e. if there are 2 channels and 18 bands per channel, 
        pyr_tensor[:,0:18,...] will contain the pyr responses for channel 1 and 
        pyr_tensor[:, 18:36, ...] will contain the responses for channel 2). 
        In the case of a complex, multichannel pyramid with split_complex=True,
        the real/imaginary bands will be intereleaved so that they appear as pairs with
        neighboring indices in the channel dimension of the tensor (Note: the residual bands are always
        real so they will only ever have a single band even when split_complex=True.)

        Parameters
        ----------
        pyr_coeffs: `OrderedDict`
            the pyramid coefficients
        split_complex: `bool`
            indicates whether the output should split complex bands into real/imag channels or keep them as a single
            channel. This should be True if you intend to use a convolutional layer on top of the output. 

        Returns
        -----------
        pyr_tensor: `torch.Tensor` (BxCxHxW)
            pyramid coefficients reshaped into tensor. The first channel will be the residual highpass and the last will be
            the residual lowpass. Each band is then a separate channel. 
        pyr_info: `List` 
            containing the number of channels, if split_complex was used
            in the convert_pyr_to_tensor, and the list of pyramid keys for the dictionary

        
        Note:conversion to tensor only works for pyramids without downsampling of feature maps 
        """
        
        pyr_keys = tuple(pyr_coeffs.keys())
        test_band = pyr_coeffs[pyr_keys[0]]
        num_channels = test_band.size(1)
        coeff_list = []
        key_list = []
        for ch in range(num_channels):
            coeff_list_resid = []
            coeff_list_bands = []
            for k in pyr_keys:
                coeffs = pyr_coeffs[k][:,ch:(ch+1),...]
                if 'residual' in k:
                    coeff_list_resid.append(coeffs)
                else:
                    if (coeffs.dtype in complex_types) and split_complex:
                        coeff_list_bands.extend([coeffs.real, coeffs.imag])
                    else:
                        coeff_list_bands.append(coeffs)

            
            if 'residual_highpass' in pyr_coeffs.keys():
                coeff_list_bands.insert(0,coeff_list_resid[0])
                if 'residual_lowpass' in pyr_coeffs.keys():
                    coeff_list_bands.append(coeff_list_resid[1])
            elif 'residual_lowpass' in pyr_coeffs.keys():
                coeff_list_bands.append(coeff_list_resid[0])

            coeff_list.extend(coeff_list_bands)
        
        try:
            pyr_tensor = torch.cat(coeff_list, dim=1)
            pyr_info = tuple([num_channels, split_complex, pyr_keys])
        except RuntimeError as e:
            raise Exception("""feature maps could not be concatenated into tensor. 
            Check that you are using coefficients that are not downsampled across scales. 
            This is done with the 'downsample=False' argument for the pyramid""")
            

        return pyr_tensor, pyr_info

