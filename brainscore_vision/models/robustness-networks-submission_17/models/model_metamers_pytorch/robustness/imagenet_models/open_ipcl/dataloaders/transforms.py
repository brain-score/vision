import math
import numpy as np
import torch
import random
import numbers
import kornia
import cv2
from PIL import Image
from IPython.core.debugger import set_trace
from torch.nn.functional import relu
from kornia.filters.kernels import get_gaussian_kernel2d
from albumentations import transforms as AT

from . import functional as F
from . import functional_tensor as FT

__all__ = ['GaussianBlurMoCo', 'RandomGaussianBlurGPU',
           'ColorJitterGPU', 'ToGrayscaleTorchGPU',
           'HSVJitterGPU', 'ToGrayscaleGPU', 'RandomGrayscaleGPU',
           'RandomBrightness', 'RandomContrastGPU',
           'ToNumpy', 'ToChannelsFirst', 'ToChannelsLast', 
           'ToDevice', 'ToFloat', 'ToFloatDiv',
           'NormalizeGPU',
           'RandomHorizontalFlipGPU', 'RandomRotateGPU', 'RandomZoomGPU',
           'RandomRotateObjectGPU',
           'Compose', 'RandomApply',
           'FixedOpticalDistortionGPU', 'CircularMaskGPU',
           'LMS_To_LGN', 'LMS_To_LGN_Lum', 'LMS_To_LGN_Color']

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================================================
#  Normalize
# =================================================

class NormalizeGPU(object):
    """x = (x - mean) / std"""

    def __init__(self, mean, std, inplace=True, device=default_device):
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)
        self.inplace = inplace
        self.device = default_device 
        
    def apply_last(self, b):
        return F.normalize(b, self.mean, self.std, self.inplace)
        
    def __call__(self, b):
        return F.normalize(b, self.mean, self.std, self.inplace)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += ', mean={0}'.format(self.mean)
        format_string += ', std={0}'.format(self.std)
        format_string += ', inplace={0})'.format(self.inplace)
        return format_string

# =================================================
#  Gaussian Blur
# =================================================

def quick_blur(x):
    sigma = random.uniform(.1, 2.0)
    size = int(math.ceil(sigma*2.5))
    if size % 2 == 0: size = size + 1
    gauss = kornia.filters.GaussianBlur2d((size, size), (sigma, sigma))
    return gauss(x)

class RandomGaussianBlurGPU(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, p=.5, sigma=[.1, 2.]):
        self.p = p
        self.sigma_range = sigma
        self.kernel = None
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        sigma = kwargs['sigma'] if 'sigma' in kwargs else sigma
        
        size = int(math.ceil(sigma*2.5))
        if size % 2 == 0: size = size + 1
        
        self.sigma = sigma if self.do else None
        self.size = size if self.do else None        
        self.kernel = kornia.filters.GaussianBlur2d((size, size), (sigma, sigma)) if self.do else None
    
    def last_params(self):
        return {
            "do": self.do,
            "sigma": self.sigma,
            "size": self.size
        }
    
    def apply_last(self, b):
        params = self.last_params()
        if len(b.shape)==3:
            return self.kernel(b.unsqueeze(0)).squeeze() if params["do"] else b            
        else:
            return self.kernel(b) if params["do"] else b
        
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        if len(b.shape)==3:
            return self.kernel(b.unsqueeze(0)).squeeze() if self.do else b            
        else:
            return self.kernel(b) if self.do else b
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', sigma={0})'.format(self.sigma_range)
        return format_string

class GaussianBlurMoCo(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
# =================================================
#  Instance Sampler
# =================================================

class InstanceSamplesTransform:
    """transform input `n_samples` times"""
    def __init__(self, base_transform, n_samples=5):
        self.base_transform = base_transform
        self.n_samples = n_samples

    def __call__(self, x):
        samples = [self.base_transform(x) for i in range(self.n_samples)]
        return samples
    
    def __repr__(self):
        text = f'{self.__class__.__name__}(n_samples={self.n_samples})\n'
        text += self.base_transform.__repr__()
        return text
    
# =================================================
#  ColorJitterGPU
# =================================================

class ColorJitterGPU(object):
    """Randomly change the hue, saturation, value (aka brightness), and contrast of an RGB image or batch.
    
    Parameters are stored for easy replay.

    Args:
        p (float): probability that jitter should be applied. If input is a batch, the p operates at the batch level
                   (i.e., either all images are jittered, or none, with probability p). Also, if jitter is applied,
                   each property is jittered by a random value in the range specified (see below for how to set 
                   range for each property).
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have -.5 < hue < .5
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-.5,.5) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        value (float or tuple of python:float (min, max)) – range over which to jitter value / brightness. 
            value_factor is chosen uniformly from [max(0, 1 - value), 1 + value] or the given [min, max]. 
            Should be non negative numbers. 
            
        contrast (float or tuple of python:float (min, max)) – range over which to jitter contrast in RGB. 
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. 
            Should be non negative numbers.  
    
    """

    def __init__(self, p=1.0, hue=0.0, saturation=0.0, value=0.0, contrast=0.0):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)    
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        self.contrast = self._check_input(contrast, 'contrast')
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "h": self.h,
            "s": self.s,
            "v": self.v,
            "c": self.c,
        }
    
    def apply_last(self, b):
        params = self.last_params()
        return FT.color_jitter(b, params['h'], params['s'], params['v'], params['c']) if params['do'] else b
    
    @staticmethod
    def sample_params(n, hue, saturation, value, contrast):
        
        if hue is not None:
            hue_factor = torch.FloatTensor(n).uniform_(hue[0], hue[1])
            #hue_factor.mul_(FT.M_PI).div_(180)
        else:
            hue_factor = None # skip adjust_hue
        
        if saturation is not None:
            saturation_factor = torch.FloatTensor(n).uniform_(saturation[0], saturation[1])
        else:
            saturation_factor = None # skip adjust_saturation
            
        if value is not None:
            value_factor = torch.FloatTensor(n).uniform_(value[0], value[1])
        else:
            value_factor = None # skip adjust_value (brightness)
        
        if contrast is not None:
            contrast_factor = torch.FloatTensor(n).uniform_(contrast[0], contrast[1])
        else:
            contrast_factor = None # skip adjust_constrast
        
        return hue_factor, saturation_factor, value_factor, contrast_factor
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        n = b.shape[0] if len(b.shape) == 4 else 1
        h,s,v,c = self.sample_params(n, self.hue, self.saturation, self.value, self.contrast)
        self.h = kwargs['h'] if 'h' in kwargs else h
        self.s = kwargs['s'] if 's' in kwargs else s
        self.v = kwargs['v'] if 'v' in kwargs else v
        self.c = kwargs['c'] if 'c' in kwargs else c
        
        if 'mask' in kwargs:
            self.mask = kwargs['mask']
            self.h[self.mask] = 0 
            self.s[self.mask] = 1
            self.v[self.mask] = 1
            self.c[self.mask] = 1
            
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            h (float or list of float): hue, overrides random draw; 
            s (float or list of float): saturation, overrides random draw
            v (float or list of float): value, overrides random draw
            c (float or list of float): contrast, overrides random draw
            mask (tensor same size as batch, boolean): True locations in mask are not jittered
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with hue, sat, val, contrast jittered with
            different random params for each individual image.
        """
        self.before_call(b, **kwargs)
        return FT.color_jitter(b, self.h, self.s, self.v, self.c) if self.do else b        
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0})'.format(self.value)
        format_string += ', contrast={0})'.format(self.contrast)
        return format_string



# =================================================
#  HSVJitterGPU
# =================================================

class HSVJitterGPU(object):
    """Randomly change the hue, saturation, and value of an RGB image using a Yiq conversion
    for computational efficiency. 

    Args:
        p (float): probability that jitter should be applied.
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have 0<= hue <= 180 or -180 <= min <= max <= 180.            
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-180,180) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        brightness (float or tuple of python:float (min, max)) – range over which to jitter brightness. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers. 
    
    """

    def __init__(self, p=.80, hue=0., saturation=0.0, value=0.0):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-180., 180.),
                                     clip_first_on_zero=False)
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "h": self.h,
            "s": self.s,
            "v": self.v
        }
    
    def apply_last(self, b):
        params = self.last_params()
        return F.hsv_jitter(b, params['h'], params['s'], params['v']) if params['do'] else b
    
    @staticmethod
    def sample_params(n, hue, saturation, value):

        if hue is not None:
            #hue_factor = random.uniform(hue[0], hue[1])
            hue_factor = torch.FloatTensor(n).uniform_(hue[0], hue[1]).numpy()
        else:
            hue_factor = 0. # rotate by 0, no change
            
        if saturation is not None:
            #saturation_factor = random.uniform(saturation[0], saturation[1])
            saturation_factor = torch.FloatTensor(n).uniform_(saturation[0], saturation[1])
        else:
            saturation_factor = 1. # multiply by 1, no change
            
        if value is not None:
            #value_factor = random.uniform(value[0], value[1])
            value_factor = torch.FloatTensor(n).uniform_(value[0], value[1])
        else:
            value_factor = 1. # multiply by 1, no change
        
        return hue_factor, saturation_factor, value_factor
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        n = b.shape[0] if len(b.shape) == 4 else 1
        h,s,v = self.sample_params(n, self.hue, self.saturation, self.value)
        self.h = kwargs['h'] if 'h' in kwargs else h
        self.s = kwargs['s'] if 's' in kwargs else s
        self.v = kwargs['v'] if 'v' in kwargs else v
        if 'mask' in kwargs:
            self.mask = kwargs['mask']
            self.h[self.mask] = 0
            self.s[self.mask] = 1
            self.v[self.mask] = 1
            
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            h (float or list of float): hue, overrides random draw; 
            s (float or list of float): saturation, overrides random draw
            v (float or list of float): value, overrides random draw
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with hue, sat, val jittered with
            different random params for each individual image.
        """
        self.before_call(b, **kwargs)
        return F.hsv_jitter(b, self.h, self.s, self.v) if self.do else b        
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0})'.format(self.value)
        return format_string

    
class HSVJitterGPU2(object):
    """Randomly change the hue, saturation, and value of an RGB image using a Yiq conversion
    for computational efficiency. 

    Args:
        p (float): probability that jitter should be applied.
        
        hue (float or tuple of python:float (min, max)) – range over which to jitter hue
            Should have 0<= hue <= 180 or -180 <= min <= max <= 180.            
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. 
            hue=0 for no change, hue=(-180,180) maximum color randomization.
            
        saturation (float or tuple of python:float (min, max)) – range over which to jitter saturation. 
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            Should be non negative numbers.
        
        value (float or tuple of python:float (min, max)) – range over which to jitter brightness in YiQ. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers. 
            
        brightness (float or tuple of python:float (min, max)) – range over which to jitter brightness in RGB. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers.            
            
        contrast (float or tuple of python:float (min, max)) – range over which to jitter brightness in RGB. 
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            Should be non negative numbers.                      
    
    """

    def __init__(self, p=.80, hue=0., saturation=0.0, value=0.0, brightness=0.0, contrast=0.0):
        self.p = p
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-180., 180.),
                                     clip_first_on_zero=False)
        self.saturation = self._check_input(saturation, 'saturation')
        self.value = self._check_input(value, 'value')
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')        
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value            
    
    def last_params(self):
        return {
            "do": self.do,
            "h": self.h,
            "s": self.s,
            "v": self.v,
            "b": self.b,
            "c": self.c
        }
    
    def apply_last(self, b):
        params = self.last_params()
        return F.hsv_jitter2(b, params['h'], params['s'], params['v'], params['b'], params['c']) if params['do'] else b
    
    @staticmethod
    def sample_params(n, hue, saturation, value, brightness, contrast):

        if hue is not None:
            #hue_factor = random.uniform(hue[0], hue[1])
            hue_factor = torch.FloatTensor(n).uniform_(hue[0], hue[1]).numpy()
        else:
            hue_factor = 0. # rotate by 0, no change
            
        if saturation is not None:
            #saturation_factor = random.uniform(saturation[0], saturation[1])
            saturation_factor = torch.FloatTensor(n).uniform_(saturation[0], saturation[1])
        else:
            saturation_factor = 1. # multiply by 1, no change
            
        if value is not None:
            #value_factor = random.uniform(value[0], value[1])
            value_factor = torch.FloatTensor(n).uniform_(value[0], value[1])
        else:
            value_factor = 1. # multiply by 1, no change
            
        if brightness is not None:
            #value_factor = random.uniform(value[0], value[1])
            brightness_factor = torch.FloatTensor(n).uniform_(brightness[0], brightness[1])
        else:
            brightness_factor = 1. # multiply by 1, no change
            
        if contrast is not None:
            #value_factor = random.uniform(value[0], value[1])
            contrast_factor = torch.FloatTensor(n).uniform_(contrast[0], contrast[1])
        else:
            contrast_factor = 1. # multiply by 1, no change
        
        return hue_factor, saturation_factor, value_factor, brightness_factor, contrast_factor
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        n = b.shape[0] if len(b.shape) == 4 else 1
        h,s,v,b,c = self.sample_params(n, self.hue, self.saturation, self.value, self.brightness, self.contrast)
        self.h = kwargs['h'] if 'h' in kwargs else h
        self.s = kwargs['s'] if 's' in kwargs else s
        self.v = kwargs['v'] if 'v' in kwargs else v
        self.b = kwargs['b'] if 'b' in kwargs else b
        self.c = kwargs['c'] if 'c' in kwargs else c
        if 'mask' in kwargs:
            self.mask = kwargs['mask']
            self.h[self.mask] = 0
            self.s[self.mask] = 1
            self.v[self.mask] = 1
            self.b[self.mask] = 1
            self.c[self.mask] = 1
            
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            h (float or list of float): hue, overrides random draw; 
            s (float or list of float): saturation, overrides random draw
            v (float or list of float): value, overrides random draw
            b (float or list of float): brightness, overrides random draw
            c (float or list of float): contrast, overrides random draw
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with hue, sat, val jittered with
            different random params for each individual image.
        """
        self.before_call(b, **kwargs)
        return F.hsv_jitter2(b, self.h, self.s, self.v, self.b, self.c) if self.do else b        
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', hue={0}'.format(self.hue)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', value={0}'.format(self.value)
        format_string += ', brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        return format_string
    
# =================================================
#  ToGrayscaleGPU
# =================================================

class ToGrayscaleGPU(object):
    """Convert image to Grayscale
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
    
    def __call__(self, b):
        return F.to_grayscale(b, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels) 

class ToGrayscaleTorchGPU(object):
    """Convert image to Grayscale
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
    
    def __call__(self, b):
        return FT.rgb_to_grayscale(b, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels) 
    
# =================================================
#  Compose
# =================================================

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
    
    def apply_last(self, img):
        for t in self.transforms:
            img = t.apply_last(img)
        return img
    
    def __call__(self, img, replay=False):
        for t in self.transforms:
            img = t.apply_last(img) if replay else t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
# =================================================
#  RandomApply
# =================================================

class RandomApply(object):
    """Randomly apply transforms with probability `p`
    """

    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p
        self.do = None
        
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        #self.do, self.idx = F.mask_batch(b, p=self.p)
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        
    def last_params(self):
        return {"do": self.do}
    
    def apply_last(self, b):
        params = self.last_params()
        if params['do']: 
            for t in self.transforms: 
                b = t.apply_last(b)
        return b
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        if self.do: 
            for t in self.transforms: 
                b = t(b)
        return b
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += ',\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
# =================================================
#  RandomGrayscaleGPU
# =================================================

class RandomGrayscaleGPU(object):
    """Convert image to Grayscale
    """

    def __init__(self, p=.5):
        self.p = p

    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.do, self.idx = F.mask_batch(b, p=self.p)
        
    def last_params(self):
        return {"do": self.do, "idx": self.idx}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.random_grayscale(b, params['idx'])
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.random_grayscale(b, self.idx)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p) 
    
# =================================================
#  RandomBrightnessGPU
# =================================================

class RandomBrightness(object):
    """Randomly Adjust Brightness with probability `p`, with scale_factor drawn
        uniformly between `scale_range`=[min,max] separately for each image in a batch,
        clamping at maximum brightness `max_value`.
        
        new_img = img * scale_factor
        
    """

    def __init__(self, p=1.0, scale_range=(.6,1.4), max_value=1.0):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
    
    def before_call(self, b, **kwargs):
        "determine whether to `do` transform, and scale_factor `sf` for each image in batch `b`"
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        sf = torch.FloatTensor(n).uniform_(self.scale_range[0], self.scale_range[1])
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        self.sf = kwargs['sf'] if 'sf' in kwargs else sf
        
    def last_params(self):
        '''get parameters from last random draw'''
        return {"do": self.do, "sf": self.sf if self.do else None}
    
    def apply_last(self, b):
        '''apply last parameters from last random draw to batch `b`'''
        params = self.last_params()
        return self.__call__(b, sf=params['sf']) if params['do'] else b
    
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            sf (float or list of float): scale_factors, overrides random draw performed in before_call; 
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with brightness adjusted.
        """        
        self.before_call(b, **kwargs)
        return F.adjust_brightness(b, scale_factor=self.sf, max_value=self.max_value) if self.do else b

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', scale_range={0}'.format(self.scale_range)
        format_string += ', max_value={0})'.format(self.max_value)
        return format_string
    
# =================================================
#  RandomContrastGPU
# =================================================    

class RandomContrastGPU(object):
    """Randomly Adjust Contrast with probability `p`, with scale_factor drawn
        uniformly between `scale_range`=[min,max] separately for each image in a batch,
        clamping at maximum brightness `max_value`.
        
        new_img = (1-scale_factor)*img.mean() - (scale_factor) * img
    """

    def __init__(self, p=1.0, scale_range=(.6,1.4), max_value=1.0):
        self.scale_range = scale_range
        self.p = p
        self.max_value = max_value
    
    def before_call(self, b, **kwargs):
        "determine whether to `do` transform, and scale_factor `sf` for each image in batch `b`"
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        sf = torch.FloatTensor(n).uniform_(self.scale_range[0], self.scale_range[1])
        self.do = any(kwargs) or self.p==1. or random.random() < self.p
        self.sf = kwargs['sf'] if 'sf' in kwargs else sf
        
    def last_params(self):
        '''get parameters from last random draw'''
        return {"do": self.do, "sf": self.sf if self.do else None}
    
    def apply_last(self, b):
        '''apply last parameters from last random draw to batch `b`'''
        params = self.last_params()
        return self.__call__(b, sf=params['sf']) if params['do'] else b
    
    def __call__(self, b, **kwargs):
        """
        Args:
            b (TensorImage or ArrayImage): Image or batch of Images to be jittered.

        kwargs (optional, to override random draw with desired parameters):
            sf (float or list of float): scale_factors, overrides random draw performed in before_call; 
            If a kwarg is passed, the transform is always performed.
            
        Returns:
            Image or batch of Images with brightness adjusted.
        """        
        self.before_call(b, **kwargs)
        return F.adjust_contrast(b, scale_factor=self.sf, max_value=self.max_value) if self.do else b

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', scale_range={0}'.format(self.scale_range)
        format_string += ', max_value={0})'.format(self.max_value)
        return format_string
    
# =================================================
#  type/device transforms
# =================================================

class ToNumpy(object):
    """Converts the given Image to a numpy array.
    """
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted.
        Returns:
            ArrayImage: image as numpy array
        """
        return F.to_numpy(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToChannelsFirst(object):
    """Converts batch from BxHxWxC to BxCxHxW
    """
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
            img (PIL Image): Image to be converted.
        Returns:
            ArrayImage: image as numpy array
        """            
        return F.to_channels_first(x)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToChannelsLast(object):
    """Converts batch from BxCxHxW to BxHxWxC
    """
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        """
        Args:
            img: ArrayImage or TensorImage to be converted.
        Returns:
            ArrayImage: image with channels_last
        """            
        return F.to_channels_last(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'    
    
class ToDevice(object):
    """Moves tensor to device.
    """
    def __init__(self, device):
        self.device = device
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.to_device(x, self.device)

    def __repr__(self):
        return self.__class__.__name__ + "(device='{0}')".format(self.device)  

class ToFloat(object):
    """Converts tensor to float using .float()
    """
    def __init__(self, value):
        self.value = float(value)
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.to_float(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToFloatDiv(object):
    """Converts tensor to float using division.
    """
    def __init__(self, value):
        self.value = float(value)
    
    def apply_last(self, x):
        return self(x)
    
    def __call__(self, x):
        return F.div_(x, val=self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(value={0})'.format(self.value)    
    
    
# =================================================
#  RandomHorizontalFlipGPU
# =================================================

class RandomHorizontalFlipGPU(object):
    """Flip the input horizontally around the y-axis with probability `p`.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with flipping deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5):
        '''`p` is the probability that each image is flipped'''
        self.p = p

    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        self.flip_val = b.new_ones(n) - 2 * self.do
        
        if 'flip_val' in kwargs:
            self.flip_val = kwargs['flip_val'].to(b.device)
            self.do = (self.flip_val == -1).float()
        self.mat = F._prepare_mat(b, F.flip_mat(self.flip_val))
        
    def last_params(self):
        return {"do": self.do, "flip_val": self.flip_val, "mat": self.mat}
    
    def apply_last(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'])
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p) 
    
    
# =================================================
#  RandomRotateGPU
# =================================================    
    
class RandomRotateGPU(object):
    """Randomly rotate image around a point.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with flipping deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5, max_deg=45, x=.5, y=.5, pad_mode='zeros'):
        '''
            `p` is the probability that each image is rotated            
        '''
        self.p = p
        self.max_deg = max_deg
        self.x_range = (x, x) if isinstance(x, (int,float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int,float)) else tuple(y)
        self.pad_mode = pad_mode
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        self.deg = b.new(n).uniform_(-self.max_deg, self.max_deg) * self.do
        self.xs = b.new(n).uniform_(self.x_range[0], self.x_range[1])
        self.ys = b.new(n).uniform_(self.y_range[0], self.y_range[1])
        self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))
        
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'deg' in kwargs: self.deg = kwargs['deg'].to(b.device)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device)
            
            self.do = (self.deg.abs() != 0).float()
            self.mat = F._prepare_mat(b, F.rotate_mat(self.deg, self.xs, self.ys))
        
    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', max_deg={0}'.format(self.max_deg)
        format_string += ', x_range={0}'.format(self.x_range)
        format_string += ', y_range={0}'.format(self.y_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string    
    
    
# =================================================
#  RandomZoomGPU
# ================================================= 

class RandomZoomGPU(object):
    """Randomly zoom image around a point.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with zoom deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations.
        
    """
        
    def __init__(self, p=.5, zoom=(.5,1.0), x=.5, y=.5, pad_mode='zeros'):
        '''
            p is the probability that each image is rotated
            x is...
            y is...
        '''
        self.p = p
        self.zoom_range = (zoom, zoom) if isinstance(zoom, (int,float)) else tuple(zoom)
        self.x_range = (x, x) if isinstance(x, (int,float)) else tuple(x)
        self.y_range = (y, y) if isinstance(y, (int,float)) else tuple(y)
        self.pad_mode = pad_mode
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        self.zoom = b.new(n).uniform_(self.zoom_range[0], self.zoom_range[1]) * self.do + (1-self.do)
        self.xs = b.new(n).uniform_(self.x_range[0], self.x_range[1])
        self.ys = b.new(n).uniform_(self.y_range[0], self.y_range[1])
        self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))
        
        return {"do": self.do, "zoom": self.zoom, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'zoom' in kwargs: self.deg = kwargs['zoom'].to(b.device)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device)
            
            self.do = (self.zoom.abs() != 1.0).float()
            self.mat = F._prepare_mat(b, F.zoom_mat(self.zoom, self.xs, self.ys))
        
    def last_params(self):
        return {"do": self.do, "zoom": self.zoom, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', zoom={0}'.format(self.zoom_range)
        format_string += ', x={0}'.format(self.x_range)
        format_string += ', y={0}'.format(self.y_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string
        
    
# =================================================
#  RandomRotateObjectGPU
# ================================================= 

class RandomRotateObjectGPU(object):
    """Randomly rotate, rescale, and re-position an object.
    
    Works for a single TensorImage or TensorBatch, on cpu or gpu,
    with parameters deteremined and applied separately for each individual image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.
    
    Targets:
        TensorImage, TensorBatch
        
        Not applied to ArrayImage or ArrayBatch because we rely on PyTorch grid_sample,
        which only works for tensors.
        
        For ArrayImage or ArrayBatch use albumentations HorizontalFlip.
        
    """
        
    def __init__(self, p=.5, max_deg=45, ctr_x=.5, ctr_y=.5, scale=(1.0,2.0), 
                 dest_x=(.25,.75), dest_y=(.25,.75), pad_mode='border'):
        '''
                  p: the probability that each image is rotated
            max_deg: maximum angle of rotation (+/- max_deg)
            
        '''
        self.p = p
        self.max_deg = max_deg
        self.cx_range = (ctr_x, ctr_x) if isinstance(ctr_x, (int,float)) else tuple(ctr_x)
        self.cy_range = (ctr_y, ctr_y) if isinstance(ctr_y, (int,float)) else tuple(ctr_y)
        self.scale_range = (scale, scale) if isinstance(scale, (int,float)) else tuple(scale)
        self.destx_range = (dest_x, dest_x) if isinstance(dest_x, (int,float)) else tuple(dest_x)
        self.desty_range = (dest_y, dest_y) if isinstance(dest_y, (int,float)) else tuple(dest_y)
        
        self.pad_mode = pad_mode
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        self.deg = b.new(n).uniform_(-self.max_deg, self.max_deg) * self.do
        self.xs = b.new(n).uniform_(self.cx_range[0], self.cx_range[1]) * self.do + (1-self.do)*.5
        self.ys = b.new(n).uniform_(self.cy_range[0], self.cy_range[1]) * self.do + (1-self.do)*.5
        self.scale = b.new(n).uniform_(self.scale_range[0], self.scale_range[1]) * self.do + (1-self.do)
        self.dest_x = b.new(n).uniform_(self.destx_range[0], self.destx_range[1]) * self.do + (1-self.do)*.5
        self.dest_y = b.new(n).uniform_(self.desty_range[0], self.desty_range[1]) * self.do + (1-self.do)*.5
        
        mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
        self.mat = F._prepare_mat(b, mat)
        
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
        if any(kwargs):
            if 'deg' in kwargs: self.deg = kwargs['deg'].to(b.device)
            if 'xs' in kwargs: self.xs = kwargs['xs'].to(b.device)
            if 'ys' in kwargs: self.ys = kwargs['ys'].to(b.device)
            if 'scale' in kwargs: self.ys = kwargs['scale'].to(b.device)
            if 'dest_x' in kwargs: self.ys = kwargs['dest_x'].to(b.device)
            if 'dest_y' in kwargs: self.ys = kwargs['dest_y'].to(b.device)
            
            self.do = (self.deg.abs() != 0).float()
            mat = F.rotate_object_mat(self.deg, self.xs, self.ys, self.scale, self.dest_x, self.dest_y)
            self.mat = F._prepare_mat(b, mat)
        
    def last_params(self):
        return {"do": self.do, "deg": self.deg, "xs": self.xs, "ys": self.ys, "mat": self.mat}
    
    def replay(self, b):
        params = self.last_params()
        return F.affine_transform(b, params['mat'], pad_mode=self.pad_mode)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.affine_transform(b, self.mat, pad_mode=self.pad_mode)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', max_deg={0}'.format(self.max_deg)
        format_string += ', ctr_x={0}'.format(self.cx_range)
        format_string += ', ctr_y={0}'.format(self.cy_range)
        format_string += ', scale={0}'.format(self.cy_range)
        format_string += ',\n\t\t      dest_x={0}'.format(self.destx_range)
        format_string += ', dest_y={0}'.format(self.desty_range)
        format_string += ", pad_mode='{0}')".format(self.pad_mode)
        return format_string
    
    
# =================================================
#  Optical Distortion GPU
# =================================================    

class FixedOpticalDistortionGPU(object):
    """Optical distortion with fixed distoration params.
    
        Works for a single TensorImage or TensorBatch, on cpu or gpu.
        
        Adapted from Albumentations OpticalTransform to operate on TensorImage and TensorBatch
        
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py
        
    """
        
    def __init__(self, width, height, distortion=-.5, dx=0, dy=0, device='cpu',
                 interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, border_value=None):
        '''
            distortion: controls fisheye/barrel transformation
            dx: shift in x direction
            dy: shift in y direction            
        '''
        self.width = width
        self.height = height
        self.p = 1.0 # always perform the transform
        self.distortion = distortion
        self.dx = dx
        self.dy = dy
        self.device = device
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.grid = self.gen_grid(distortion, dx, dy, width, height)
        
    def gen_grid(self, k, dx, dy, width, height):
        '''relying on cv2 to compute the remapping
            - not sure if this would be fast enough to compute separate maps for each image
              in a batch, but for our purposes we're applying the same transform to all
              images in the batch, so this wont be an issue...
        '''
        H, W = self.height, self.width

        fx = W
        fy = H

        cx = width * 0.5 + dx
        cy = height * 0.5 + dy

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (W, H), cv2.CV_32FC1)
        
        # normalize the map [-1, 1]
        nmap1 = torch.tensor( (map1 - W/2) / (W/2) ).float()
        nmap2 = torch.tensor( (map2 - H/2) / (H/2) ).float()
        
        # store as grid to pass to grid_sample
        grid = torch.stack([nmap1,nmap2],dim=2).unsqueeze(0).to(self.device)

        #img = cv2.remap(img, map1, map2, interpolation=self.interpolation, borderMode=self.border_mode, 
        #                borderValue=self.border_value)
        
        return grid
    
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        
        # grid is fixed
        # self.grid = self.gen_grid(k, dx, dy, width, height)
        
        return {"do": self.do, "grid": self.grid}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
    def last_params(self):
        return {"do": self.do, "grid": self.grid}
    
    def replay(self, b):
        params = self.last_params()
        return F.grid_sample(b, params['grid'], align_corners=False)
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.grid_sample(b, self.grid, align_corners=False)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', distortion={0}'.format(self.distortion)
        format_string += ', dx={0}'.format(self.dx)
        format_string += ', dy={0}'.format(self.dy)
        format_string += ', width={0}'.format(self.width)
        format_string += ', height={0}'.format(self.height)
        format_string += ", device='{0}')".format(self.device)
        return format_string
    
# =================================================
#  Circular Mask
# =================================================     

def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return (x, y)

class CircularMaskGPU(object):
    """Apply a circular mask to each image.
    
        Works for a single TensorImage or TensorBatch, on cpu or gpu.
    """
        
    def __init__(self, image_size, blur_span=24.0, tol=.0005, device='cpu'):
        
        self.p = 1.0
        self.image_size = image_size
        self.blur_span = blur_span
        self.blur_radius = image_size/2 - blur_span
        self.tol = tol
        self.blank = Image.new("RGB", (image_size,image_size), 0)
        self.device = device
        self._gen_mask()
    
    def _gen_mask(self):
        blur_span = torch.tensor(self.blur_span).float()
        zero = torch.tensor(0.).float()
        
        # get the distance of each pixel from the center
        vals = torch.tensor(range(self.image_size)).float()
        x,y = torch.meshgrid(vals, vals)
        rho, phi = cart2pol(x-vals.mean(),y-vals.mean())
        
        # create a normal distribution with std that scales from `max` to
        # zero (within `tol`) over the span `blur_span`
        scale = blur_span
        normal = torch.distributions.normal.Normal(loc=0, scale=scale)
        v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()
        while v > self.tol:
            scale = scale*.975
            normal = torch.distributions.normal.Normal(loc=0, scale=scale)
            v = normal.log_prob(blur_span).exp() / normal.log_prob(zero).exp()
        
        # compute the mask value, adjusting each rho value to be distance from blur_radius
        # and then using the normal distribution to drop contrast graduate from zero beyond that
        # radius; afterwards reset everything within blur radius to 1.0
        self.mask = normal.log_prob(rho-self.blur_radius).exp() / normal.log_prob(zero).exp()
        self.mask[rho < self.blur_radius] = 1.0
        self.pil_mask = Image.fromarray( (self.mask*255).numpy().astype(np.uint8))
        self.mask = self.mask.view(1,1,*self.mask.shape).to(self.device)
        
    def draw_params(self, b):
        n = b.shape[0] if (hasattr(b, 'shape') and len(b.shape) == 4) else 1
        
        self.do = F.mask_tensor(b.new_ones(n), p=self.p)
        
        # mask is fixed
        
        return {"do": self.do, "mask": self.mask}
    
    def before_call(self, b, **kwargs):
        "before_call the state for input `b`"
        self.draw_params(b)
        
    def last_params(self):
        return {"do": self.do, "mask": self.mask}
    
    def replay(self, b):
        params = self.last_params()
        return F.apply_mask(b, params['mask'])
    
    def __call__(self, b, **kwargs):        
        self.before_call(b, **kwargs)
        return F.apply_mask(b, self.mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('  
        format_string += 'p={0}'.format(self.p)
        format_string += ', image_size={0}'.format(self.image_size)
        format_string += ', blur_span={0}'.format(self.blur_span)
        format_string += ', tol={0}'.format(self.tol)
        format_string += ", device='{0}')".format(self.device)
        return format_string
    
# =================================================
#  LMS + LGN Transforms
# =================================================      

class SRGB_to_LMS(object):

    def __call__(self, b, replay=False):
        if len(b.shape)==4:
            x = F.srgb_to_lrgb(b)
        elif len(b.shape)==3:
            x = F.srgb_to_lrgb(b.unsqueeze(0)).squeeze()
        else:
            raise ValueError("input should be BxCxHxW or CxHxW, Got {b.shape}")
        
        return x

    def __repr__(self):
        return self.__class__.__name__+'()'
    
class LMS_To_LGN(object):
    
    def __init__(self, size=21, sigma=2.5, sigmaRatio=0.5, polarity=0,
                 oppSize=21, oppCenterSigma=2.5, oppSurroundSigma=2.5*1.57,
                 pad_mode='reflect', device=None, lm_only=False):
        super().__init__()
        
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.pad_mode = pad_mode
        self.lm_only = lm_only
        
        # luminance channel, difference of gaussians filter
        self.size = size
        self.sigma = sigma
        self.sigmaRatio = sigmaRatio                
        self.polarity = polarity # 0 => off-center unit
        self.kernel1 = get_gaussian_kernel2d((size,size),(sigma,sigma)).unsqueeze(0)
        self.kernel2 = get_gaussian_kernel2d((size,size),(sigma*sigmaRatio,sigma*sigmaRatio)).unsqueeze(0)
        self.DoG = (self.kernel1 - self.kernel2) if polarity == 0 else (self.kernel2 - self.kernel1)
        
        # center, surround gaussians for computing single-opponent color responses
        self.oppSize = oppSize
        self.oppCenterSigma = oppCenterSigma
        self.oppSurroundSigma = oppSurroundSigma
        self.oppCenter = get_gaussian_kernel2d((oppSize,oppSize),(oppCenterSigma,oppCenterSigma)).unsqueeze(0)
        self.oppSurround = get_gaussian_kernel2d((oppSize,oppSize),(oppSurroundSigma,oppSurroundSigma)).unsqueeze(0)
        
    def __call__(self, x, replay=False):
        
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        
        # luminance can be over l+m responses only
        if self.lm_only:
            luminance = x[:,0:2,:,:].sum(dim=1,keepdims=True)
        else:
            luminance = x[:,0:3,:,:].sum(dim=1,keepdims=True)
        OffCellResponse = kornia.filter2D(luminance, self.DoG, self.pad_mode)
        OnCellResponse = -1.0 * OffCellResponse
        OffCellResponse[OffCellResponse < 0] = 0
        OnCellResponse[OnCellResponse < 0] = 0
        
        # single opponent, center minus surround
        c = kornia.filter2D(x, self.oppCenter, self.pad_mode)
        s = kornia.filter2D(x, self.oppSurround, self.pad_mode)
        Lp_Mn = c[:,0:1,:,:] - s[:,1:2,:,:] # L+/M-
        Mp_Ln = c[:,1:2,:,:] - s[:,0:1,:,:] # M+/L-
        Sp_LMn = c[:,2:3,:,:] - (s[:,0:1,:,:]*.5 + s[:,1:2,:,:]*.5) # S+/(L+M)- with L+M averaged
        LMp_Sn = (c[:,0:1,:,:]*.5 + c[:,1:2,:,:]*.5) - s[:,2:3,:,:] # (L+M)+/S-
                
        out = torch.cat([OnCellResponse, OffCellResponse, 
                         Lp_Mn, Mp_Ln, 
                         Sp_LMn, LMp_Sn], dim=1)
 
        return relu(out.squeeze())

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += ', sigma={0:3.3f}'.format(self.sigma)
        format_string += ', sigmaRatio={0:3.3f}'.format(self.sigmaRatio)
        format_string += ', oppSize={0}'.format(self.oppSize)
        format_string += ', oppCenterSigma={0:3.3f}'.format(self.oppCenterSigma)
        format_string += ', oppSurroundSigma={0:3.3f}'.format(self.oppSurroundSigma)
        format_string += f", device='{self.device}'"
        format_string += ')'
        return format_string
    
class LMS_To_LGN_Lum(object):
    
    def __init__(self, size=21, sigma=2.5, sigmaRatio=0.5, polarity=0,
                 pad_mode='reflect', device=None, lm_only=False):
        super().__init__()
        
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.pad_mode = pad_mode
        self.lm_only = lm_only
        
        # luminance channel, difference of gaussians filter
        self.size = size
        self.sigma = sigma
        self.sigmaRatio = sigmaRatio                
        self.polarity = polarity # 0 => off-center unit
        self.kernel1 = get_gaussian_kernel2d((size,size),(sigma,sigma)).unsqueeze(0)
        self.kernel2 = get_gaussian_kernel2d((size,size),(sigma*sigmaRatio,sigma*sigmaRatio)).unsqueeze(0)
        self.DoG = (self.kernel1 - self.kernel2) if polarity == 0 else (self.kernel2 - self.kernel1)
        
    def __call__(self, x, replay=False):
        
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        
        # should be over L+M only?
        if self.lm_only:
            luminance = x[:,0:2,:,:].sum(dim=1,keepdims=True)
        else:
            luminance = x[:,0:3,:,:].sum(dim=1,keepdims=True)
        OffCellResponse = kornia.filter2D(luminance, self.DoG, self.pad_mode)
        OnCellResponse = -1.0 * OffCellResponse
        OffCellResponse[OffCellResponse < 0] = 0
        OnCellResponse[OnCellResponse < 0] = 0
                
        out = torch.cat([OnCellResponse, OffCellResponse], dim=1)
        
        return relu(out.squeeze())

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += ', sigma={0:3.3f}'.format(self.sigma)
        format_string += ', sigmaRatio={0:3.3f}'.format(self.sigmaRatio)
        format_string += f", device='{self.device}'"
        format_string += ')'
        return format_string
    
class LMS_To_LGN_Color(object):
    
    def __init__(self, oppSize=21, oppCenterSigma=2.5, oppSurroundSigma=2.5*1.57,
                 pad_mode='reflect', device=None):
        super().__init__()
        
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.pad_mode = pad_mode

        # center, surround gaussians for computing single-opponent color responses
        self.oppSize = oppSize
        self.oppCenterSigma = oppCenterSigma
        self.oppSurroundSigma = oppSurroundSigma
        self.oppCenter = get_gaussian_kernel2d((oppSize,oppSize),(oppCenterSigma,oppCenterSigma)).unsqueeze(0)
        self.oppSurround = get_gaussian_kernel2d((oppSize,oppSize),(oppSurroundSigma,oppSurroundSigma)).unsqueeze(0)
        
    def __call__(self, x, replay=False):
        
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        
        # single opponent, center minus surround
        c = kornia.filter2D(x, self.oppCenter, self.pad_mode)
        s = kornia.filter2D(x, self.oppSurround, self.pad_mode)
        Lp_Mn = c[:,0:1,:,:] - s[:,1:2,:,:] # L+/M-
        Mp_Ln = c[:,1:2,:,:] - s[:,0:1,:,:] # M+/L-
        Sp_LMn = c[:,2:3,:,:] - (s[:,0:1,:,:]*.5 + s[:,1:2,:,:]*.5) # S+/(L+M)- with L+M averaged
        LMp_Sn = (c[:,0:1,:,:]*.5 + c[:,1:2,:,:]*.5) - s[:,2:3,:,:] # (L+M)+/S-
                
        out = torch.cat([Lp_Mn, Mp_Ln, 
                         Sp_LMn, LMp_Sn], dim=1)
        
        return relu(out.squeeze())

    def __repr__(self):
        format_string = self.__class__.__name__ 
        format_string += ', oppSize={0}'.format(self.oppSize)
        format_string += ', oppCenterSigma={0:3.3f}'.format(self.oppCenterSigma)
        format_string += ', oppSurroundSigma={0:3.3f}'.format(self.oppSurroundSigma)
        format_string += f", device='{self.device}'"
        format_string += ')'
        return format_string 
        
# =================================================
#  Custom Albumentation Style Transforms
# ================================================= 

class CenterCropResize(AT.DualTransform):
    """CenterCropResize
    
        Center Crop and Resize a HxWxC Numpy Array.
        
    """
    def __init__(self, width, height, crop_width, crop_height, 
                 interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        '''
            width, height: final output height in pixels
            crop_width, crop_height: pct of original width and height to crop
        '''
        super(CenterCropResize, self).__init__(always_apply, p)
        #self.p = p
        self.width = width
        self.height = height
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.interpolation = interpolation
        #self.always_apply = always_apply
        
#         self.h_start = None
#         self.w_start = None
#         self._additional_targets = {}
        
#         # replay mode params
#         self.deterministic = False
#         self.save_key = "replay"
#         self.params = {}
#         self.replay_mode = False
#         self.applied_in_replay = False
        
    def get_params_dependent_on_targets(self, params):        
        img = params["image"]
        h,w,c = img.shape
        return {
            "h_start": ((h - self.crop_height)/2)/h,
            "w_start": ((w - self.crop_width)/2)/w,
            "crop_height": self.crop_height,
            "crop_width": self.crop_width,
        }
    
    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "height", "width", "crop_height", "crop_width", "interpolation"
    
    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, 
              interpolation=cv2.INTER_LINEAR, **params):
        crop = F.crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)
    
# class CenterCropResize(object):
#     """CenterCropResize
    
#         Center Crop and Resize a HxWxC Numpy Array.
        
#     """
#     def __init__(self, out_width, out_height, crop_width, crop_height, interpolation=cv2.INTER_LINEAR):
#         '''
            
#         '''
#         self.out_width = out_width
#         self.out_height = out_height
#         self.crop_width = crop_width
#         self.crop_height = crop_height
#         self.interpolation = interpolation
        
#         self.h_start = None
#         self.w_start = None
        
#     def draw_params(self, b):
#         h,w,c = b.shape
#         self.h_start = ((h - self.crop_height)/2)/h
#         self.w_start = ((w - self.crop_width)/2)/w

#     def before_call(self, b, **kwargs):
#         "before_call the state for input `b`"
#         self.draw_params(b)

#     def last_params(self):
#         return {
#             "h_start": self.h_start, 
#             "w_start": self.w_start, 
#             "crop_height": self.crop_height, 
#             "crop_width": self.crop_width
#         }
    
#     def replay(self, b):
#         params = self.last_params()
#         crop = F.crop(self.crop_height, self.crop_width, self.h_start, self.w_start)
#         return F.resize(crop, self.height, self.width, self.interpolation)
    
#     def __call__(self, b, **kwargs):  
#         self.before_call(b, **kwargs)
#         crop = F.crop(b, self.crop_height, self.crop_width, self.h_start, self.w_start)
#         return F.resize(crop, self.out_height, self.out_width, self.interpolation)
    
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('  
#         format_string += 'out_width={0}'.format(self.out_width)
#         format_string += ', out_height={0}'.format(self.out_height)
#         format_string += ', crop_width={0}'.format(self.crop_width)
#         format_string += ', crop_height={0}'.format(self.crop_height)
#         format_string += ', interpolation={0})'.format(self.interpolation)
#         return format_string    