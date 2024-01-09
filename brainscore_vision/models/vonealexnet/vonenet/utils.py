import numpy as np
import torch


def gabor_kernel(frequency, sigma_x, sigma_y, theta=0, offset=0, ks=61):
    w = ks // 2
    grid_val = torch.arange(-w, w + 1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = torch.zeros(y.shape)
    g[:] = torch.exp(-0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= torch.cos(2 * np.pi * frequency * rotx + offset)

    return g


def sample_dist(hist, bins, ns, scale="linear"):
    rand_sample = np.random.rand(ns)
    if scale == "linear":
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == "log2":
        rand_sample = np.interp(
            rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins)
        )
        rand_sample = 2**rand_sample
    elif scale == "log10":
        rand_sample = np.interp(
            rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins)
        )
        rand_sample = 10**rand_sample
    return rand_sample
