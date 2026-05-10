"""Utilities for adding, merging, and splitting temporal dimensions."""

import torch
from torch import nn

class AddTemporalDim(nn.Module):
    """Add a temporal dimension to the input tensor."""
    def __init__(self, T: int) -> None:
        """Initialize the temporal expander.

        Args:
            T: Number of time steps to repeat.
        """
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor):
        """Expand input by repeating along a new temporal dimension.

        Args:
            x: Input tensor.

        Returns:
            Tensor with a leading temporal dimension.
        """
        return x.unsqueeze(0).repeat(self.T, *torch.ones(x.ndim, dtype=torch.int64))

class MergeTemporalDim(nn.Module):
    """Merge temporal dimension into the batch dimension."""
    def __init__(self, T: int) -> None:
        """Initialize the temporal merger.

        Args:
            T: Number of time steps.
        """
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor):
        """Merge temporal and batch dimensions.

        Args:
            x: Input tensor with temporal dimension.

        Returns:
            Tensor with merged batch dimension.
        """
        return x.flatten(0, 1).contiguous()

class SplitTemporalDim(nn.Module):
    """Split the merged batch dimension back into time and batch."""
    def __init__(self, T: int) -> None:
        """Initialize the temporal splitter.

        Args:
            T: Number of time steps.
        """
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor):
        """Split merged dimension into temporal and batch dimensions.

        Args:
            x: Input tensor with merged batch dimension.

        Returns:
            Tensor with leading temporal dimension.
        """
        y_shape = [self.T, int(x.shape[0] / self.T)]
        y_shape.extend(x.shape[1:])
        return x.view(y_shape)
