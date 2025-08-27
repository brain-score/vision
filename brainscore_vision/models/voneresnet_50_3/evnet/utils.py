import math
import torch
import numpy as np
import random

def gaussian_kernel(
    sigma: float, k: float=1, size:float=15, norm:bool=False
    ) -> torch.tensor:
    """Returns a 2D Gaussian kernel.

    :param sigma (float): standard deviation of the Gaussian
    :param k (float, optional): height of the Gaussian
    :param size (float, optional): kernel size
    :param norm (bool, optional): whether no normalize the kernel
    :return: gaussian kernel
    """
    assert size % 2 == 1
    w = size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    gaussian = k * torch.exp(-(x**2 + y**2) / (2*(sigma)**2))
    if norm: gaussian /= torch.abs(gaussian.sum())
    return gaussian


def dog_kernel(
    sigma_c: float, sigma_s: float, k_c: float, k_s: float,
    polarity:int, size:int=21
    ) -> torch.tensor:
    """Returns a 2D Difference-of-Gaussians kernel.

    :param sigma_c: standard deviation of the center Gaussian
    :param sigma_s: standard deviation of the surround Gaussian
    :param k_c: peak sensitivity of the center
    :param k_s: peak sensitivity of the surround
    :param polarity: polarity of the center Gaussian (+1 or -1)
    :param size: kernel size
    :return: difference-of-gaussians kernel
    """
    assert size % 2 == 1
    assert polarity in [-1 , 1]
    center_gaussian = gaussian_kernel(sigma=sigma_c, k=k_c, size=size)
    surround_gaussian = gaussian_kernel(sigma=sigma_s, k=k_s, size=size)
    dog = polarity * (center_gaussian - surround_gaussian)
    dog /= torch.sum(dog)
    return dog

def circular_kernel(size:int, radius:float) -> torch.tensor:
    """Returns circular kernel.

    :param size (int): kernel size
    :param radius (float): radius of the circle
    :return: circular kernel
    """

    w = size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    kernel = torch.zeros(y.shape)
    kernel[torch.sqrt(x**2 + y**2) <= radius] = 1
    kernel /= torch.sum(kernel)
    return kernel

def gabor_kernel(
    frequency:float,  sigma_x:float, sigma_y:float,
    theta:float=0, offset:float=0, ks:int=61
    ):
    """Returns gabor kernel.

    :param frequency (float): spatial frequency of gabor
    :param sigma_x (float): standard deviation in x direction
    :param sigma_y (float): standard deviation in y direction
    :param theta (int, optional): Angle theta. Defaults to 0.
    :param offset (int, optional): Offset. Defaults to 0.
    :param ks (int, optional): Kernel size. Defaults to 61.
    :return: np.ndarray: 2-dimensional Gabor kernel
    """
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

def generate_grating(
    size:int, radius:float, sf:float, theta:float=0, phase:float=0,
    contrast:float=1, gaussian_mask:bool=False
    ) -> torch.tensor:
    """Returns masked grating array.

    :param size (int): kernel size
    :param radius (float): standard deviation times sqrt(2) of the mask if gaussian_mask is True, and the radius if is false
    :param sf (float): spatial frequency of the grating
    :param theta (float, optional): angle of the grating 
    :param phase (float, optional): phase of the grating
    :param gaussian_mask (bool, optional): mask is a Gaussian if true and a circle if false 
    :param contrast (float, optional): maximum contrast of the grating
    :return: 2d masked grating array
    """
    grid_val = torch.linspace(-size//2, size//2+1, size, dtype=torch.float)
    X, Y = torch.meshgrid(grid_val, grid_val, indexing='ij')
    grating = torch.sin(2*math.pi*sf*(X*math.cos(theta) + Y*math.sin(theta)) + phase) * contrast
    mask = torch.exp(-((X**2 + Y**2)/(2*(radius/np.sqrt(2))**2))) if gaussian_mask else torch.sqrt(X**2 + Y**2) <= radius
    return grating * mask * .5 + .5


def sample_dist(hist:np.array, bins:int, ns:float, scale:str='linear'):
    """Samples from distributions with different scales.

    Args:
        hist (np.array): histogram
        bins (int): number of bins
        ns (float): sample size
        scale (str, optional): distribution scale. Defaults to 'linear'.

    :returns rand_sample (np.array): 
    """    
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

def set_seed(seed):
    """Enforces deterministic behaviour and sets RNG seed for numpy and pytorch.

    :param seed (int): seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

