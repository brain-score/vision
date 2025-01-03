import torch
from torch import nn
import numpy as np
import cv2
from numpy.typing import NDArray

class CurvatureFilters(nn.Module):
    """
    A module for generating curvature filters used to process images.
    
    Attributes:
        n_ories (int): Number of orientations for the filters.
        in_channels (int): Number of input channels.
        curves (np.ndarray): Array of curvature values for the filters.
        gau_sizes (tuple): Sizes of the Gaussian windows.
        filt_size (int): Size of the filter (width and height).
        fre (list): List of spatial frequencies to be used in the filters.
        gamma (float): Aspect ratio of the filter.
        sigx (float): Sigma value along x-axis for the Gaussian envelope.
        sigy (float): Sigma value along y-axis for the Gaussian envelope.
    """
    def __init__(self, n_ories: int = 16, in_channels: int = 1, curves: NDArray[np.float64] = np.logspace(-2, -0.1, 5),
                 gau_sizes=(5,), filt_size: int = 9, fre=[1.2], gamma: float = 1, sigx: float = 1, sigy: float = 1) -> None:
        super().__init__()
        self.n_ories = n_ories
        self.curves = curves
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy
        self.in_channels = in_channels

    def forward(self):
        """
        forward pass through the module.
        """
        i = 0
        ories = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.curves) * len(self.gau_sizes) * len(self.fre), self.in_channels, self.filt_size, self.filt_size))
        for curve in self.curves:
            for gau_size in self.gau_sizes:
                for orie in ories:
                    for f in self.fre:
                        w[i, 0, :, :] = banana_filter(gau_size, f, orie, curve, self.gamma, self.sigx, self.sigy, self.filt_size)
                        i += 1
        return w        

    
def banana_filter(s: float, fre: float, theta: float, cur: float, gamma: float, sigx: float, sigy: float, sz: int) -> torch.Tensor:
    """
    Creates a single curvature filter (banana-shaped) using specified parameters.

    Args:
        s (float): Scale factor for the filter.
        fre (float): Spatial frequency of the filter.
        theta (float): Orientation angle of the filter.
        cur (float): Curvature of the filter.
        gamma (float): Aspect ratio of the filter.
        sigx (float): Sigma value along x-axis.
        sigy (float): Sigma value along y-axis.
        sz (int): Size of the filter.

    Returns:
        torch.Tensor: The generated filter as a 2D tensor.
    """
    # Define a matrix that used as a filter
    xv, yv = np.meshgrid(np.arange(np.fix(-sz/2).item(), np.fix(sz/2).item() + sz % 2),
                         np.arange(np.fix(sz/2).item(), np.fix(-sz/2).item() - sz % 2, -1))
    xv = xv.T
    yv = yv.T

    # Define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # Define the bias term
    bias = np.exp(-sigx / 2)
    k = xc + cur * (xs ** 2)

    # Define the rotated Guassian rotated and curved function
    k2 = (k / sigx) ** 2 + (xs / (sigy * s)) ** 2
    G = np.exp(-k2 * fre ** 2 / 2)

    # Define the rotated and curved complex wave function
    F = np.exp(fre * k * 1j)

    # Multiply the complex wave function with the Gaussian function with a constant and bias
    filt = gamma * G * (F - bias)
    filt = np.real(filt)
    filt -= filt.mean()

    filt = torch.from_numpy(filt).float()
    return filt


class GaborFilters(nn.Module):
    """
    A module for generating Gabor filters used to process images .
    
    Parameters:
        n_ories (int): Number of orientations for the filters.
        in_channels (int): Number of input channels for each filter.
        filt_size (int): Dimension (width and height) of each square filter.
        num_scales (int): Number of scales to generate filters for.
        min_scale (int): Minimum scale of the filters.
        max_scale (int): Maximum scale of the filters.
    """
    
    def __init__(self, n_ories: int = 12, in_channels: int = 1, filt_size: int = 5, num_scales: int = 3, min_scale: int = 5, max_scale: int = 15) -> None:
        super().__init__()
        self.n_ories = n_ories
        self.filt_size = filt_size
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self) -> torch.Tensor:
        """
        forward pass through the module
        """
        orientations = np.linspace(0, np.pi, self.n_ories, endpoint=False)
        scales = np.linspace(self.min_scale, self.max_scale, self.num_scales)
        w = torch.zeros((self.n_ories * self.num_scales, self.in_channels, self.filt_size, self.filt_size))

        i = 0
        for scale in scales:
            for orientation in orientations:
                sigma = 1
                theta = orientation
                lambda_ = scale / np.pi
                psi = 0
                gamma = 0.5

                w[i, 0, :, :] = torch.Tensor(cv2.getGaborKernel((self.filt_size, self.filt_size), sigma, theta, lambda_, gamma, psi))
                i += 1
    
        return w


def filters(in_channels: int, filter_type: str, filter_params: dict, kernel_size: int) -> torch.Tensor:
    """
    Gneerates specified types of filters as PyTorch tensors.
    
    Parameters:
        in_channels (int): Number of input channels for the filters.
        filter_type (str): Type of filter to generate ('curvature' or 'gabor').
        filter_params (dict): Parameters specific to the type of filter to generate.
        kernel_size (int): Size of the filters (width and height).

    Returns:
        torch.Tensor: A tensor containing the requested filters.

    Raises:
        AssertionError: If the filter type provided is not supported.
    """
    
    assert filter_type in ['curvature', 'gabor'], "Filter type not found, it should 'curvature' or  'gabor'"

    if filter_type == 'curvature':
        return CurvatureFilters(
            in_channels=in_channels,
            n_ories=filter_params['n_ories'],
            gau_sizes=filter_params['gau_sizes'],
            curves=np.logspace(-2, -0.1, filter_params['n_curves']),
            fre=filter_params['spatial_fre'],
            filt_size=kernel_size
        )()
    elif filter_type == 'gabor':
        return GaborFilters(
            in_channels=in_channels,
            n_ories=filter_params['n_ories'],
            num_scales=filter_params['num_scales'],
            filt_size=kernel_size
        )()
