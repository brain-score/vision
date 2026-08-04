"""Convolution layers."""

from typing import Any, Union

import numpy as np
import torch
from torch import nn

__all__ = [
    'SpikingConv2d', 'SpikingEiConv2d'
]


class SpikingConv2d(nn.Module):
    """2D convolution wrapper."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1, bias: bool = False,
                 padding_mode: str = 'zeros'):
        """Initialize module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding size.
            dilation: Dilation factor.
            groups: Number of blocked connections.
            bias: Whether to add a bias term.
            padding_mode: Padding mode for convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias, padding_mode)

        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = True

    def _set_visualize_cache(self, *args: torch.Tensor) -> None:
        """Cache tensors for visualization."""
        with torch.no_grad():
            inputs, output = args
            self.visualize_cache['param1:weight'] = self.conv.weight.detach()
            self.visualize_cache['data1:input'] = inputs.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution to the input.

        Args:
            inputs: Input tensor.

        Returns:
            Convolved output tensor.
        """
        output = self.conv.forward(inputs)
        if self._need_visualize:
            self._set_visualize_cache(inputs, output)
        return output

    def set_visualize(self, flag: bool) -> None:
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag


class SpikingEiConv2d(nn.Module):
    """E-I 2D convolution with dynamic initialization, 
    including E-to-E and E-to-I paths."""

    def __init__(self, in_channels: int, out_channels: int, ei_ratio: int,
                 device: torch.device, rng: np.random.Generator,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1, bias: bool = False,
                 padding_mode: str = 'zeros') -> None:
        """Initialize module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of excitatory output channels.
            ei_ratio: # E neurons / # I neurons.
            device: Device for parameter allocation.
            rng: Random generator for initialization.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding size.
            dilation: Dilation factor.
            groups: Number of blocked connections.
            bias: Whether to add a bias term.
            padding_mode: Padding mode for convolution.
        """
        super().__init__()
        self.n_e = out_channels
        self.n_i = self.n_e // ei_ratio
        self.rng = rng
        self.device = device

        # E-to-E conv
        self.conv_ee = nn.Conv2d(in_channels, self.n_e, kernel_size, stride,
                                 padding, dilation, groups, bias, padding_mode,
                                 device=self.device)
        # E-to-I conv
        self.conv_ie = nn.Conv2d(in_channels, self.n_i, kernel_size, stride,
                                 padding, dilation, groups, bias, padding_mode,
                                 device=self.device)
        
        self.in_features = int(np.prod(self.conv_ee.weight.shape[1:]))
        self.exp_scale: float | None = None

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = False

    def _dynamic_init(self, batch_stats: dict[str, Any]) -> None:
        """Initialize weights based on input batch statistics."""
        with torch.no_grad():
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.exp_scale = np.sqrt(Var_x / (self.in_features * (E_x_square + Var_x))) # Eq. 53 in paper

            weight_ee_np = self.rng.exponential(
                scale=self.exp_scale,
                size=self.conv_ee.weight.shape,
            )
            self.conv_ee.weight.data = torch.from_numpy(weight_ee_np).float().to(self.device)

            weight_ie_np = self.rng.exponential(
                scale=self.exp_scale,
                size=self.conv_ie.weight.shape,
            )
            self.conv_ie.weight.data = torch.from_numpy(weight_ie_np).float().to(self.device)

    def _get_batch_stats(self, x: torch.Tensor) -> dict[str, Any]:
        """Compute summary statistics of the input batch.

        Args:
            x: Input tensor.

        Returns:
            Dictionary of batch statistics.
        """
        with torch.no_grad():
            batch_stats = {}
            batch_stats['E_x'] = x.mean().item()
            batch_stats['Var_x'] = x.var(dim=0).mean().item()
            batch_stats['E_x_square'] = (x ** 2).mean().item()
            return batch_stats

    def _set_visualize_cache(self, *args: torch.Tensor) -> None:
        """Cache tensors for visualization."""
        with torch.no_grad():
            inputs, I_ee, I_ie = args
            self.visualize_cache['param1:weight_ee'] = self.conv_ee.weight.detach()
            self.visualize_cache['param2:weight_ie'] = self.conv_ie.weight.detach()
            self.visualize_cache['data1:input'] = inputs.detach()
            self.visualize_cache['data2:I_ee'] = I_ee.detach()
            self.visualize_cache['data3:I_ie'] = I_ie.detach()

    def _clamp_parameters(self) -> None:
        """Clamp parameters to be non-negative."""
        with torch.no_grad():
            self.conv_ee.weight.data.clamp_(min=0)
            self.conv_ie.weight.data.clamp_(min=0)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Union[dict, None]]:
        """Apply E-I convolution and return currents with stats.

        Args:
            inputs: Input tensor.

        Returns:
            Tuple of excitatory current, inhibitory current, and batch stats.
        """
        self._clamp_parameters()

        batch_stats = None
        if not self._inited:
            batch_stats = self._get_batch_stats(inputs)
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ee = self.conv_ee.forward(inputs)
        I_ie = self.conv_ie.forward(inputs)
        if self._need_visualize:
            self._set_visualize_cache(inputs, I_ee, I_ie)
        return I_ee, I_ie, batch_stats

    def set_visualize(self, flag: bool) -> None:
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag
