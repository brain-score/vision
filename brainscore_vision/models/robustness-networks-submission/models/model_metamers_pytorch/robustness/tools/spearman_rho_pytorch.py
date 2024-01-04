import numpy as np
import torch

def _find_repeats(x):
    unique, counts = torch.unique(x, return_counts=True)
    return unique[counts > 1]


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp, dtype=torch.float32).cuda()
    ranks[tmp] = torch.arange(x.shape[0], dtype=torch.float32).cuda()

    repeats = _find_repeats(x)
    for r in repeats:
        location_repeat = (x==r)
        ranks[location_repeat] = torch.mean(ranks[location_repeat])
    return ranks

def _get_cov_1d_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    return 1/(x.shape[0]-1) * torch.matmul(x-mu_x, y-mu_y)

def spearman_correlation_pytorch(x: torch.Tensor, y: torch.Tensor):
    """Compute spearman correlation between 2 1-D tensors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    cov_xy = _get_cov_1d_tensors(x_rank, y_rank)
    var_x = torch.sqrt(torch.var(x_rank))
    var_y = torch.sqrt(torch.var(y_rank))
    return cov_xy / (var_x * var_y)
