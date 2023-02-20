import torch.nn as nn
import torch as ch
from .custom_modules import FakeReLUM, FakeReLU
from .layers import conv2d_same, pool2d_same
import sys
sys.path.append('/om/user/jfeather/python-packages/spectrotemporal-filters')
import spectemp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

__all__ = ['spectemp_filts_time_average_coch1']

class SincWithKaiserWindowTimeOnly(nn.Module):
    """
    Makes a downsampling window that is a sinc function windowed with a kaiser window.
    """
    def __init__(self, sr, downsample_factor, window_size=21):
        super(SincWithKaiserWindowTimeOnly, self).__init__()
        self.sr = sr
        self.window_size = window_size
        self.downsample_factor = int(downsample_factor)
        downsample_filter_np = self._make_downsample_filter_response()
        self.register_buffer('downsample_filter', ch.from_numpy(downsample_filter_np).float())
        
    def forward(self, x): # TODO: implement different padding
        # TODO: is this the fastest way to apply the weighted average?
        # https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840/2
        x_shape = x.shape
        if len(x_shape)>3:
            x = x.view(x_shape[0]*x_shape[1]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0]*x_shape[1], 1, -1)
        x = ch.nn.functional.conv1d(x, self.downsample_filter, 
                                    stride=self.downsample_factor)
        return x.view(x_shape[0:-1] + (-1,))

    def _make_downsample_filter_response(self):
        downsample_filter_times = np.arange(-self.window_size/2,
                                            int(self.window_size/2))
        downsample_filter_response_orig = (np.sinc(downsample_filter_times / 
                                                  self.downsample_factor) /
                                              self.downsample_factor)
        downsample_filter_window = signal.kaiser(self.window_size, 5)
        downsample_filter_response = (downsample_filter_window * 
                                         downsample_filter_response_orig)
        downsample_filter_response = np.expand_dims(np.expand_dims(
                                         downsample_filter_response, 0), 0)

#         return ch.Tensor(downsample_filter_response)
        return downsample_filter_response


class SpecTempFilterBank(nn.Module):
    """
    Spectro-Temporal filterbanks that are a model of primary auditory cortex. Inspired by Chi-Ru-Shamma 2002. 
    """
    def __init__(self):

        super(SpecTempFilterBank, self).__init__()

        self.use_rfft = True

        # TODO: HOW TO GET THE SIZE OF THE COCHLEAGRAM
        # TODO: Make these inputs to the function
        self.N_T = 390
        self.N_F = 211     
        self.ENV_SR = 200
        self.SIGNAL_LENGTH_S = 2
        self.N_F = 211 # Set this to be the number of frequency channels in the cochleagram
        self.sr_erb = 6 # Approximatey correct... #TODO: do full calculation
        self.sr_Hz = 200
        self.make_plots = False
        # Temporal modulation rates for the filters
        self.temp_mod_rates = [0, 0.5, 1, 2, 4, 8, 16, 32, 64]
        # Spectral modulation rates for the filters
        self.spec_mod_rates = [0, 0.0625, 0.125, 0.25, 0.5, 1, 2]

        # Pad the cochleagram in frequency and in time (temporal, spectral)
        self.zero_pad = (self.N_T*2,self.N_F)
#         self.zero_pad = (0,0)

        # Make the lowest filter lowpass and the highest filter highpass (so that the full modulation spectrum is covered)
        self.low_and_highpass=True

        # all_Hts is the fourier representation for each filter. This gets multiplied with the fourier representation
        # of the cochleagram.
        # spec_temp_freqs stores all of the filter frequency values
        self.all_Hts_numpy, self.spec_temp_freqs = spectemp.make_Hts_filterbank(self.N_F,self.N_T, self.sr_Hz, self.sr_erb,
                                                               temp_mod_rates=self.temp_mod_rates,
                                                               spec_mod_rates=self.spec_mod_rates,
                                                               make_plots=self.make_plots,
                                                               zero_pad=self.zero_pad,
                                                               low_and_highpass=self.low_and_highpass)
        self.all_Hts_numpy = np.float32(self._split_real_imaginary(self.all_Hts_numpy))

        # (pad_left, pad_right, pad_top, pad_bottom)
        self.padding = nn.ConstantPad2d((0, self.zero_pad[0], 0, self.zero_pad[1]), 0)
        # TODO: Make the apply filter step a pytorch module so that it gets registered appropriately!!!
        self.apply_filt = ApplyFiltInFourier2D(self.all_Hts_numpy, self.N_F, self.N_T,
                                               use_rfft=self.use_rfft, 
                                               zero_pad=self.zero_pad,
                                               )
        self.square_real = SquareReal()
#         self.rectify = HalfWaveRectify()
        # Downsample the power by a factor of 2
#         self.downsample_power = SincWithKaiserWindowTimeOnly(self.ENV_SR, 2)

#       No need to take the complex abs because the imaginary part is zero.
#         self.compute_power = ComplexAbsoluteValuePower()

    def forward(self, x, with_latent=False, fake_relu=False):
        if with_latent:
            all_outputs = {}

        if sum(self.zero_pad)!=0:
            x = self.padding(x)
            if with_latent:
                all_outputs['padded_input'] = x

        x = self.apply_filt(x)
        if with_latent:
            # Only save the real part of the signal -- the imaginary part is all zeros. 
            all_outputs['filtered_signal'] = x[:,:,:,:,0]

#        x = self.compute_power(x)

#         if fake_relu and with_latent:
#             all_outputs['rectified_subbands_fake_relu'] = self.rectify(x, fake_relu=fake_relu)
#         x = self.rectify(x)
#         if with_latent:
#             all_outputs['rectified_subbands'] = x

        x = self.square_real(x)
        if with_latent:
            all_outputs['spectempfilter_power'] = x

#         x = self.downsample_power(x)
#         if with_latent:
#             all_outputs['downsampled_power'] = x 

        if with_latent:
            return x, None, all_outputs
        else:
            return x

    # This is used for pytorch <1.8, before complex muliplication was used
    def _split_real_imaginary(self, complex_input):
        stacked_output = np.stack([np.real(complex_input), np.imag(complex_input)], axis=-1)
        return stacked_output


class HalfWaveRectify(nn.Module):
    """Takes the real component of the signal and applies a relu"""
    def __init__(self):
        super(HalfWaveRectify, self).__init__()
        self.fake_relu_op = FakeReLUM()

    def forward(self, x, fake_relu=False):
        real1, imag1 = [a.squeeze(-1) for a in ch.split(x, 1, dim=-1)]
        if fake_relu:
            return self.fake_relu_op(real1)
        else:
            return ch.nn.functional.relu(real1)


class ApplyFiltInFourier2D(nn.Module):
    """
    Spectro-Temporal filterbanks that are a model of primary auditory cortex. Inspired by Chi-Ru-Shamma 2002.
    """
    def __init__(self, all_Hts, N_F, N_T, use_rfft=True, zero_pad=(0,0)):
        super(ApplyFiltInFourier2D, self).__init__()
        self.register_buffer('all_Hts', ch.tensor(all_Hts))
        self.N_F = N_F
        self.N_T = N_T
        self.use_rfft = use_rfft
        self.zero_pad = zero_pad

    # TODO: update to work with pytorch>1.8 which uses complex multiplication
    def forward(self, x):
        if self.use_rfft:
            x = x.squeeze(dim=1)
            x_fft = ch.rfft(x, 2, onesided=False).unsqueeze(1) # Add channel dim
        else:
            x = x.squeeze(dim=1)
            # Add imaginary component for the full fft
            x = ch.stack([x, ch.zeros(x.shape)], dim=-1)
            x_fft = ch.fft(x, 2).unsqueeze(1) # Add channel dim

        # Apply the filter to the signal
        filtered_signal = self._complex_multiplication2D(x_fft, self.all_Hts)

        # After filtering we might not have a purely real signal so we have to use ifft
        filtered_signal = ch.ifft(filtered_signal, 2)

        filtered_signal = filtered_signal[:,:,:self.N_F,:self.N_T]
       
        return filtered_signal

    def _complex_multiplication2D(self, t1, t2):
        real1, imag1 = [a.squeeze(-1) for a in ch.split(t1, 1, dim=-1)]
        real2, imag2 = [a.squeeze(-1) for a in ch.split(t2, 1, dim=-1)]
        return ch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

class SquareReal(nn.Module):
    """Take the real part of a signal and square it to get the power"""
    def __init__(self):
        super(SquareReal, self).__init__()
 
    def forward(self, t1):
        real1 = t1[:,:,:,:,0]
#         real1, imag1 = [a.squeeze(-1) for a in ch.split(t1, 1, dim=-1)]
        return ch.clamp(real1**2, min=1e-8)
#         return real1

class StableStd(nn.Module):
    """Computes the standard deviation over time for a signal"""
    def __init__(self, dim=-1):
        super(StableStd, self).__init__()
        self.dim = dim
    def forward(self, x):
        x_mean = ch.unsqueeze(ch.mean(x, self.dim), -1)
        N = x.shape[self.dim]
        num_sum = ch.sum((x - x_mean)**2, self.dim) 
        return ch.sqrt(ch.clamp(num_sum, 1e-16)/N)

class ComplexAbsoluteValuePower(nn.Module):
    """Complex absolute value of a signal in pytorch<1.8 (where real and imaginary parts are the last dim)
    """
    def __init__(self):
        super(ComplexAbsoluteValuePower, self).__init__()
    
    def forward(self, t1):
        real1, imag1 = [a.squeeze(-1) for a in ch.split(t1, 1, dim=-1)]
        return (real1**2 + imag1**2)

class SpectTempFilts(nn.Module):
    """Pytorch model that takes in a cochleagram as input, applies the spectrotemporal
    filters, and then takes the average of the filter power and (if specified) a linear eval
    """
    def __init__(self, coch_size, time_average=False, num_classes=1000, 
                 linear_eval=False, coch_rescale_factor=100):
        """
        Args: 
            coch_size (list of ints): shape of the input coch. Needed for the filter specification
            time_average (bool): If true, average only across time. If false, then includes an 
                average across the whole channel so thatwe can still do a linear eval
            num_classes (int): number of classes used for the linear eval
            linear_eval (bool): if true, adds a linear readout on top of the averaged layer
            coch_rescale_factor (int): rescales the cochleagram on input. Helpful if the 
                audio input representation has all low values, because many of the filters 
                will have correspondingly low output values which can be close to 
                numerical precision. 
        """
        super(SpectTempFilts, self).__init__()

        self.spectempfilterbank = SpecTempFilterBank()
        self.time_average = time_average
        self.coch_size = coch_size
        self.num_spectemp_filts = self.spectempfilterbank.all_Hts_numpy.shape[0]
        self.linear_eval = linear_eval
        self.coch_rescale_factor = coch_rescale_factor
        if self.time_average:
            self.avgpool = nn.AdaptiveAvgPool2d((None, 1)) # Average across time, not frequency
#             self.avgpool = StableStd(dim=-1)
            self.fc_input_size = coch_size[0] * self.num_spectemp_filts
        else: # Pool over everything
            self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # Average across both time and frequency
            self.fc_input_size = self.num_spectemp_filts

        if self.linear_eval:
            self.flatten = nn.Flatten()
            if isinstance(num_classes, dict):
                all_fc_layers = {}
                for task in num_classes.keys():
                    all_fc_layers[task] = nn.Linear(self.fc_input_size, num_classes[task])
                self.classification = nn.ModuleDict(all_fc_layers)
            else:
                self.classification = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        del no_relu
        
        all_outputs = {}
        all_outputs['input_after_preproc'] = x

        x = x * self.coch_rescale_factor
        all_outputs['coch_rescaled'] = x
       
        if with_latent:
            x, _, outputs_st = self.spectempfilterbank(x, with_latent=with_latent, fake_relu=fake_relu)
            all_outputs.update(outputs_st)
        else:
            x = self.spectempfilterbank(x)    

        x = self.avgpool(x)
        if with_latent:
            all_outputs['avgpool'] = x

        x_latent = x 

        if self.linear_eval:
            x_latent = self.flatten(x_latent)
            if isinstance(self.classification, nn.ModuleDict):
                final = {}
                for task, fc_l in self.classification.items():
                    final[task] = fc_l(x_latent)
            else:
                final = self.classification(x_latent)
            if with_latent:
                all_outputs['final'] = final
        else:
            final = x 

        if with_latent:
            return final, x_latent, all_outputs
        return final
        
       
def spectemp_filts_time_average_coch1(pretrained=False, **kwargs):
    """Spectrotemporal filterbank that acts on the cochleagram representation
    Includes a time average layer at the end

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    del pretrained # model checkpoint is not on model zoo
    # TODO: include the default spectemp parameters for this model
    coch_size=(211,390)
    model = SpectTempFilts(coch_size, time_average=True, linear_eval=True, **kwargs)
        
    return model

### Code to make the spectemp filters ### 
### implement the spectral temporal functions from sam norman-haignere's code (based on the NSL toolbox from shamma lab)
### Use cycles/ERB rather than cycles/octave as the way that we define the filters, because we do not interpolate the filters to a log scale. 
### Copied here on 8/27/2021 so that all components of the spectemp model are in one place. 

## implement the spectral temporal functions from sam norman-haignere's code (based on the NSL toolbox from shamma lab. 
## jfeather 3/23/17

import numpy as np
import matplotlib.pyplot as plt

def filt_temp_mod(fc_Hz, N, sr_Hz, LOWPASS=False, HIGHPASS=False, show_plots=False, zero_pad=False):
    """
    Make temporal modulation filters.
    Derived from gen_cort.m in shamma nsl toolbox and from filt_temp_mod.m in spectrotemporal-synthesis-v2 from sam norman-haignere
    
    Parameters
    ----------
    fc_Hz : int
        center frequency for the filters
    N : int
        number of temporal samples, ie how long the filter should be
    sr_Hz : int
        temporal sampling rate of the signal 
    LOWPASS : boolean
        set whether the lowest filters are lowpass (True) or bandpass (False, default). 
    HIGHPASS : boolean
        set whether the highest filters are highpass (True) or bandpass (False, default).
    show_plots : boolean
        show plots of the filters for debugging purposes (default=False, no plots) # TODO implement plots 
    zero_pad : False or int
        adds zero_pad to one_side size of zero padding (resulting in a sinc interpolation in the fourier domain) to avoid circular convolution via the fft
    
    Returns
    -------
    H : numpy array
        the frequency response of the temporal filter
    """

    if fc_Hz==0: # in case we want to use a delta for the transfer function
        H = np.zeros(N)
        if zero_pad:
            H = np.pad(H, (0, zero_pad), 'constant', constant_values=(0,0))
        H[0] = 1
        return H

    # irf
    t = np.arange(0,N)/float(sr_Hz) * float(fc_Hz) # time * fc_Hz
    h = np.sin(2.*np.pi*t) * t**2 * np.exp(-3.5*t) # gammatone

    # These are sam's lowpass filters ... they are not the same as the ones used in the NSL toolbox. 
#     if LOWPASS:
#         h = (t).^2.*exp(-3.5*t) # use a different filter if lowpass
#     else: 
#         h = np.sin(2.*np.pi*t) * t**2 * np.exp(-3.5*t) # gammatone

    # remove mean
    h = h-np.mean(h)

    if zero_pad:
        h = np.pad(h, (0, zero_pad), 'constant', constant_values=(0,0))
        N = N+zero_pad # add it here for later

    # magnitude and phase of the gammatone
    H0 = np.fft.fft(h) # TODO ask ray about fftw and how to install it. 
    th = np.angle(H0)
    A = np.abs(H0)
    A/=max(A)

    if LOWPASS | HIGHPASS:
        # nyquist if present otherwise highest pos. frequency? 
        if N%2:
            nyq_idx = int((N-1)/2) # this is to get the largest positive frequency
        else:
            nyq_idx = int(N/2)
       
        #  For an even number of input points, A[n/2] represents both positive and negative Nyquist frequency, and is also purely real for real input. 
        # index of pos/neg frequencies with max amplitude
        maxA_index_pos = np.argmax(A[0:nyq_idx+1]) # this groups the nyquist with the negative frequencies for even numbers. this is fine as long as its consistent. 
        maxA_index_neg = np.argmax(A[nyq_idx+1:]) + nyq_idx + 1 # so this is only the negative frequencies. shifted by the nyquish (+1 so it is consistent for even and odd)

        # This wont be 1 if the filter is centered on the nyquist ... could include nyquist in both? 
        # check the max is 1
        if not np.argmax(A)==nyq_idx: 
            assert all(np.isclose(A[[maxA_index_neg, maxA_index_pos]],1))
            
        # for lowpass condition
        if LOWPASS:
            A[0:maxA_index_pos] = 1
            A[maxA_index_neg:-1] = 1
        
        if HIGHPASS:
            A[maxA_index_pos:maxA_index_neg] = 1 #this should only set the nyquist to 1 if it is the max.  
    
    H = A * np.exp(np.sqrt(complex(-1))*th)
    
    return H

def filt_spec_mod(fc_cycPerb, N, sr_erb, LOWPASS=False, HIGHPASS=False, show_plots=False, zero_pad=False, GABOR=False):
    """
    Make spectral modulation filters.
    Derived from gen_corf.m in shamma nsl toolbox and from filt_spec_mod.m in spectrotemporal-synthesis-v2 from sam norman-haignere
    
    Parameters
    ----------
    fc_cycPerb : int
        center frequency (cycles per erb) of the spectral filters
    N : int
        number of spectral samples, ie how long the filter should be
    sr_erb : int
        spectral sampling rate of the cochleagram
    LOWPASS : boolean
        set whether the lowest filters are lowpass (True) or bandpass (False, default). 
    HIGHPASS : boolean
        set whether the highest filters are highpass (True) or bandpass (False, default).
    show_plots : boolean
        show plots of the filters for debugging purposes (default=False, no plots) # TODO implement plot
    zero_pad : False or int
        add equivilant zero padding of length zero_pad to one side (resulting in a sinc interpolation in the fourier domain) to avoid circular convolution via the fft    
    Returns
    -------
    H : numpy array
        the frequency response of the spectral filter
    """

    if fc_cycPerb==0: # in case we want to use a delta for the transfer function
        H = np.zeros(N)
        if zero_pad:
            H = np.pad(H, (0, zero_pad), 'constant', constant_values=(0,0))
        H[0] = 1
        return H
    
    if zero_pad:
        N = N+zero_pad
    
    # index of the nyquist if present or the maximum positive frequency. 
    if N%2:
        nyq_idx = int((N-1)/2) # this is to get the largest positive frequency
    else:
        nyq_idx = int(N/2)
        
    pos_freqs = float(sr_erb)*np.arange(0,nyq_idx+1)/float(N) # this includes the nyquist. 
    
    # check that center frequency is below the maximum positive frequency
    if fc_cycPerb>pos_freqs[nyq_idx]:
        print("Error in filt_spec_mod: center frequency exceeds the nyquist")
        return np.nan
   
    # EMAILED SAM ABOUT THIS ON 2/15/18 
    # transfer function for positive frequencies
    # if LOWPASS: # Use a gaussian if lowpass? Seems like the original code uses a gaussian if it is a lowpass filter. # TODO ask sam 
    #     per = 1./abs(fc_cycPerb)
    #     sig = per/2. # period = 2*sigma
    #     a = 1./sig**2
    #     H_pos_freqs = np.exp(-(np.pi**2)*pos_freqs**2/a) #  see: http://mathworld.wolfram.com/FourierTransformGaussian.html 
    # else: # else, keep it the same as before (gabor)
    # R1 = (pos_freqs/abs(fc_cycPerb))**2
    # H_pos_freqs = R1 * np.exp(1.-R1) # THIS IS SLIGHTLY DIFFERENT THAN SAMS CODE???? (is it the wrong side of the filter?)
        # H_pos_freqs = R1 * np.exp(-R1) # This is the same as sam's code 2/15/18
    if (GABOR) and (not LOWPASS and not HIGHPASS):
        R1 = (pos_freqs/abs(fc_cycPerb))**2
        C1 = 1/2/.3/.3
        H_pos_freqs = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)    
    elif LOWPASS: # 2/15/18 Sam updated the lowpass filters to make them a little less weird looking in the temporal domain. 
        per = 1./abs(fc_cycPerb)
        sig = per/2. # period = 2*sigma
        a = 1./sig**2
        H_pos_freqs = np.exp(-(np.pi**2)*pos_freqs**2/a) #  see: http://mathworld.wolfram.com/FourierTransformGaussian.html 
    else: 
        R1 = (pos_freqs/abs(fc_cycPerb))**2
        H_pos_freqs = R1 * np.exp(-R1) # Modifications from Sam's code ... was 1-R1, so need to renormalize below. 

    # normalize the maximum frequency # ADDED 2/15/18
    # We changed the exponent to be just -R1 instead of 1-R1. This means that the maximum will not necessarily be 1. 
    # We just renormalize it here so that the scale is correct. 
    H_pos_freqs = H_pos_freqs/max(H_pos_freqs)

    # for lowpass condition
    if LOWPASS:
        maxA_index_pos = np.argmax(H_pos_freqs)
        H_pos_freqs[0:maxA_index_pos] = 1
    # for highpass contion
    if HIGHPASS:
        maxA_index_pos = np.argmax(H_pos_freqs)
        H_pos_freqs[maxA_index_pos:] = 1
        
    # negative frequencies
    if N%2:
#         H_neg_freqs = np.conj(np.flip(H_pos_freqs[1:nyq_idx+1],0))
        H_neg_freqs = np.conj(H_pos_freqs[1:nyq_idx+1][::-1]) # duplicate the last positive frequency
    else: 
#         H_neg_freqs = np.conj(np.flip(H_pos_freqs[1:nyq_idx1],0))
        H_neg_freqs = np.conj(H_pos_freqs[1:nyq_idx][::-1]) # do not duplicate nyquist
      
    # combine the positive and negative frequency components. 
    H = np.hstack([H_pos_freqs, H_neg_freqs])
    return H

def filt_spectemp_mod(fc_cycPerb, fc_Hz, N_F, N_T, sr_erb, sr_Hz, LOWPASS_F=False, LOWPASS_T=False, HIGHPASS_F=False, HIGHPASS_T=False, show_plots=False, zero_pad=(False, False)):
    """
    Make 2D tranfer function for a spectrotemporal filter.
    Derived from filt_spectemp_mod.m in spectrotemporal-synthesis-v2 from sam norman-haignere
    
    Parameters
    ----------
    fc_cycPerb : int
        center frequency (cycles per erb) of the spectral filters.  Can be a nan if we want a frequency response of all 1s. 
    fc_Hz : int
        center frequency for the filters. Setting this positive or negative changes the directionality of the filters. Can be a nan if we want a frequency response of all 1s. 
    N_F : int
        number of spectral samples, ie how long the filter should be
    N_T : int
        number of temporal samples, ie how long the filter should be
    sr_erb : int
        spectral sampling rate of the cochleagram
    sr_Hz : int
        temporal sampling rate of the signal 
    LOWPASS_F : boolean
        set whether the lowest spectral filters are lowpass (True) or bandpass (False, default). 
    LOWPASS_T : boolean
        set whether the lowest temporal filters are lowpass (True) or bandpass (False, default).   
    HIGHPASS_F : boolean
        set whether the highest spectral filters are highpass (True) or bandpass (False, default).    
    HIGHPASS_T : boolean
        set whether the highest temporal filters are highpass (True) or bandpass (False, default).
    show_plots : boolean
        show plots of the filters for debugging purposes (default=False, no plots) # TODO implement plots
    zero_pad : list of False or int 
        sets the zero padding for the temporal (dim 1) and spectral (dim 2). If false no zero padding, and if an int adds zero padding of length zero_pad to each side 
    
    Returns
    -------
    Hts : numpy array
        the frequency response of the spectraltemporal filter. This will be zeros in two quadrants based on the sign of fc_Hz.  
    """
    
    # temporal filter
    if ~np.isnan(fc_Hz):
        Ht = filt_temp_mod(abs(fc_Hz), N_T, sr_Hz, LOWPASS_T, HIGHPASS_T, show_plots, zero_pad[0])
    else: 
        if zero_pad[0]:
            Ht = np.ones(N_T+zero_pad[0])
        else:
            Ht = np.ones(N_T)
    
    # spectral filter
    if ~np.isnan(fc_cycPerb):
        Hs = filt_spec_mod(fc_cycPerb, N_F, sr_erb, LOWPASS_F, HIGHPASS_F, show_plots, zero_pad[1])
    else:
        if zero_pad[1]:
            Hs = np.ones(N_F+zero_pad[1])
        else:                 
            Hs = np.ones(N_F)
    
    # outer product to get the spectrotemporal filter in the frequency domain
    Hts = np.outer(Ht,Hs)

    if zero_pad[0]:
        N_T = N_T+zero_pad[0]
    if zero_pad[1]:
        N_F = N_F+zero_pad[1]
    
    if N_F%2:
        nyq_idx_F = int((N_F-1)/2) # this is to get the largest positive frequency
    else:
        nyq_idx_F = int(N_F/2)
        
    if N_T%2:
        nyq_idx_T = int((N_T-1)/2) # this is to get the largest positive frequency
    else:
        nyq_idx_T = int(N_T/2)
        
    if ~np.isnan(fc_Hz) & ~np.isnan(fc_cycPerb):
    # get FFT frequencies excluding DC and nyquist
        f_spec = np.fft.fftfreq(N_F)
        f_spec[nyq_idx_F] = np.nan
        f_temp = np.fft.fftfreq(N_T)
        f_temp[nyq_idx_T] = np.nan

        
        f_spec = np.outer(np.ones(N_T), f_spec)
        f_temp = np.outer(f_temp, np.ones(N_F))

        if fc_Hz == 0: # If fc_Hz==0 then just take the quadrants as if it = 1
            first_quad_to_zero = (np.sign(f_temp)==-1) & (np.sign(f_spec)==1)
            second_quad_to_zero = (np.sign(f_temp)==1) & (np.sign(f_spec)==-1)
        else:
            first_quad_to_zero = (np.sign(f_temp)==-np.sign(fc_Hz)) & (np.sign(f_spec)==1)
            second_quad_to_zero = (np.sign(f_temp)==np.sign(fc_Hz)) & (np.sign(f_spec)==-1)
        
        Hts[first_quad_to_zero | second_quad_to_zero] = 0

    return Hts
                          
def impulse_response_spectemp(Hts, N_F, N_T):
    """
    Make 2D impulse response of the spectrotemporal filter Hts.
    The frequency response is multiplied by an impulse at the center of the time and frequency space to correct for the phase
    
    Parameters
    ----------
    Hts : numpy array
        the frequency response of the spectraltemporal filter. This will be zeros in two quadrants based on the sign of fc_Hz.  
    N_F : int
        number of spectral samples, ie how long the filter should be
    N_T : int
        number of temporal samples, ie how long the filter should be
        
    Return
    -------
    Its : numpy array
        the impulse repsponse of the spectrotemporal filters specified by Hts. 
    
    """

    # Make the impulse
    impulse_time = np.zeros([N_F,N_T])
    # center the impulse in time and frequency
    impulse_time[int(np.ceil(N_F/2.)),int(np.ceil(N_T/2.))]=1
    impulse_freq = np.fft.fft2(impulse_time)
    # perform the ifft on Hts
    Its = np.fft.ifft2((impulse_freq*Hts.T))
    return Its

def make_Hts_filterbank(N_F, N_T, sr_Hz, sr_erb, temp_mod_rates=[0,0.5,1,2,4,8,16,32,64,128], spec_mod_rates=[0,0.25,0.5,1,2,4,8], make_plots=False, db_scale=True, zero_pad=(False,False), low_and_highpass=[False,False]):
    """
    Create a spectrotemporal filterbank in frequency domain containing the specified temporal and spectroal modulation rates
    
    Parameters
    ----------
    N_F : int
        number of spectral samples, ie how long the filter should be
    N_T : int
        number of temporal samples, ie how long the filter should be
    sr_Hz : int
        temporal sampling rate of the signal     
    sr_erb : int
        spectral sampling rate of the cochleagram
    temp_mod_rates : list
        all of the temporal modulation rates to include in the filterbank
    spec_mod_rates : list
        all of the spectral modulation rates to include in the filterbank
    make_plots : boolean
        if True plots the frequency and time response for each filter in the filterbank. 
    zero_pad : list of False or int 
        sets the zero padding for the temporal (dim 1) and spectral (dim 2). If false no zero padding, and if an int adds zero padding of length zero_pad to each side 
    low_and_highpass : boolean
        if true, uses low and high pass filters for the lowest and highest (respectively) temporal and spectral modulations. defaults to false. First dimension is lowpass, second dimension is highpass. 
    
    Returns
    -------
    all_Hts : numpy array
        filterbank containig the frequency domain representation of the filters
    spec_temp_freqs : numpy array
        list of the temporal (column 0) and spectral (column 1) modulation rates for the filters in all_Hts
        
    """
    if type(low_and_highpass)==bool: # if there is only one argument, duplicate it to both fields.
        low_and_highpass = [low_and_highpass, low_and_highpass]
    # make sure that we are making the correct structure sizes if we are zero padding
    if zero_pad[1]:
        N_F_fill = N_F+zero_pad[1]
    else:
        N_F_fill = N_F
    
    if zero_pad[0]:
        N_T_fill = N_T+zero_pad[0]
    else:
        N_T_fill = N_T

    # If we have filters with 0 Hz, only make them for one value of fc_sign. 
    num_filters = len(temp_mod_rates)*len(spec_mod_rates) + (sum(np.array(temp_mod_rates)!=0)*sum(np.array(spec_mod_rates)!=0))
    all_Hts = np.zeros([num_filters,N_F_fill, N_T_fill], dtype=np.complex)
    spec_temp_freqs = np.zeros([num_filters,2])
    hts_idx = 0
    for fc_sign in (-1, 1):
        if make_plots:
            plt.figure(figsize=(30,30))
        for fc_cycPerb in spec_mod_rates:
            if (fc_sign==1) and (fc_cycPerb==0):
                continue
            if low_and_highpass[0] and (fc_cycPerb==min(spec_mod_rates)):
                LOWPASS_F = True
                HIGHPASS_F = False
            elif (fc_cycPerb==max(spec_mod_rates)) and low_and_highpass[1]:
                LOWPASS_F = False
                HIGHPASS_F = True
            else: 
                LOWPASS_F = False
                HIGHPASS_F = False
            for fc_Hz in temp_mod_rates:
                # If we have fc_Hz==0, then only run it once (not twice). 
                if (fc_sign==1) and (fc_Hz==0):
                    continue
                if low_and_highpass[0] and (fc_Hz==min(temp_mod_rates)):
                    LOWPASS_T = True
                    HIGHPASS_T = False
                elif (fc_Hz==max(temp_mod_rates)) and low_and_highpass[1]:
                    LOWPASS_T = False
                    HIGHPASS_T = True
                else:
                    LOWPASS_T = False
                    HIGHPASS_T = False
                Hts = filt_spectemp_mod(fc_cycPerb, fc_sign*fc_Hz, N_F, N_T, sr_erb, sr_Hz, zero_pad=zero_pad, LOWPASS_F=LOWPASS_F,HIGHPASS_F=HIGHPASS_F,LOWPASS_T=LOWPASS_T,HIGHPASS_T=HIGHPASS_T)
                all_Hts[hts_idx,:,:] = Hts.T
                spec_temp_freqs[hts_idx,0] = fc_sign*fc_Hz
                spec_temp_freqs[hts_idx,1] = fc_cycPerb
                if make_plots:
                    if fc_sign == -1:
                        num_spec_plots = len(spec_mod_rates)
                        num_temp_plots = len(temp_mod_rates)
                        plot_idx = hts_idx + 1
                    elif fc_sign == 1:
                        num_spec_plots = sum(np.array(spec_mod_rates)!=0)
                        num_temp_plots = sum(np.array(temp_mod_rates)!=0)
                        plot_idx = hts_idx - len(spec_mod_rates)*len(temp_mod_rates) + 1
                    plt.subplot(num_spec_plots, num_temp_plots, plot_idx)
                    if db_scale: 
                        plt.imshow(20*np.log10(np.fft.fftshift(np.abs(all_Hts[hts_idx,:,:]))),cmap='inferno',clim=[0,-20])
                    else: 
                        plt.imshow(np.fft.fftshift(np.abs(all_Hts[hts_idx,:,:])),cmap='inferno')
                    plt.colorbar()
                    plt.title('Temp Mod %.1f, Spec Mod %.1f'%(fc_sign*fc_Hz, fc_cycPerb))
                hts_idx+=1
                
    if make_plots: # make a second plot for the time domain representation of the filters
        hts_idx = 0
        for fc_sign in (-1, 1):
            plt.figure(figsize=(30,30))
            for fc_cycPerb in spec_mod_rates:
                if (fc_sign==1) and (fc_cycPerb==0):
                    continue
                for fc_Hz in temp_mod_rates:
                    if (fc_sign==1) and (fc_Hz==0):
                        continue
                    Its = impulse_response_spectemp(all_Hts[hts_idx,:,:].T, N_F_fill, N_T_fill)
                    if fc_sign == -1:
                        num_spec_plots = len(spec_mod_rates)
                        num_temp_plots = len(temp_mod_rates)
                        plot_idx = hts_idx + 1
                    elif fc_sign == 1:
                        num_spec_plots = sum(np.array(spec_mod_rates)!=0)
                        num_temp_plots = sum(np.array(temp_mod_rates)!=0)
                        plot_idx = hts_idx - len(spec_mod_rates)*len(temp_mod_rates) + 1
                    plt.subplot(num_spec_plots, num_temp_plots, plot_idx)
                    plt.imshow(np.real(Its),cmap='inferno')
                    plt.title('Temp Mod %.1f, Spec Mod %.1f'%(fc_sign*fc_Hz, fc_cycPerb))
                    hts_idx+=1
                  
    return all_Hts, spec_temp_freqs

def make_temporalmod_filterbank(N_F, N_T, sr_Hz, temp_mod_rates=[0,0.5,1,2,4,8,16,32,64,128], make_plots=False, db_scale=True, zero_pad=False):
    """
    Create a temporal modulation filterbank in frequency domain containing the specified temporal modulation rates
    
    Parameters
    ----------
    N_T : int
        number of temporal samples, ie how long the filter should be
    sr_Hz : int
        temporal sampling rate of the signal     
    temp_mod_rates : list
        all of the temporal modulation rates to include in the filterbank
    make_plots : boolean
        if True plots the frequency and time response for each filter in the filterbank. 
    zero_pad : list of False or int 
        sets the zero padding for the temporal dimension. If false no zero padding, and if an int adds zero padding of length zero_pad to each side 
    
    Returns
    -------
    all_Ht : numpy array
        filterbank containig the frequency domain representation of the filters
    temp_mod_rates : list
        all of the temporal modulation rates to include in the filterbank        
        
    """
    # make sure that we are making the correct structure sizes if we are zero padding
    
    if zero_pad:
        N_T_fill = N_T+zero_pad
    else:
        N_T_fill = N_T
     
    nyquist_T = sr_Hz/2.
    all_Ht = np.zeros([len(temp_mod_rates), N_T_fill], dtype=np.complex)
    hts_idx = 0
    if make_plots:
        plt.figure(figsize=(10,10))
    for fc_Hz in temp_mod_rates:
        if fc_Hz>nyquist_T:
            raise ValueError('At least one of the requested temporal modulation frequencies exceeds the nyquist limit for envelope sampling')
        Ht = filt_temp_mod(fc_Hz, N_T, sr_Hz, zero_pad=zero_pad)
        all_Ht[hts_idx,:] = Ht.T

        hts_idx+=1

    if make_plots:
        if db_scale: 
            plt.plot(20*np.log10(np.fft.fftshift(np.abs(all_Ht).T)))
        else: 
            plt.plot((np.fft.fftshift(np.abs(all_Ht).T)))
        
    if make_plots: # make a second plot for the time domain representation of the filters
        ht_idx = 0
        plt.figure(figsize=(10,40))
        for fc_Hz in temp_mod_rates:
            It = impulse_response_temp(all_Ht[ht_idx,:].T, N_T_fill)
            # It = np.fft.ifft(all_Ht[ht_idx,:].T)
            plt.subplot(len(temp_mod_rates), 1, ht_idx+1)
            plt.plot(np.real(It))
            plt.title('Temp Mod %.1f'%(fc_Hz))
            ht_idx+=1
                  
    return all_Ht, temp_mod_rates

def impulse_response_temp(Ht, N_T):
    """
    Make 1D impulse response of the temporal filter Hts.
    The frequency response is multiplied by an impulse at the center of the time to correct for the phase
    
    Parameters
    ----------
    Hts : numpy array
        the frequency response of the spectraltemporal filter. This will be zeros in two quadrants based on the sign of fc_Hz.  
    N_T : int
        number of temporal samples, ie how long the filter should be
        
    Return
    -------
    It : numpy array
        the impulse repsponse of the temporal modulation filter specified by Ht. 
    
    """

    # Make the impulse
    impulse_time = np.zeros([N_T])
    # center the impulse in time and frequency
    impulse_time[int(np.ceil(N_T/2.))]=1
    impulse_freq = np.fft.fft(impulse_time)
    # perform the ifft on Hts
    It = np.fft.ifft((impulse_freq*Ht.T))
    return It
