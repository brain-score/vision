"""Normalization layers for 1D inputs."""

from typing import Any, Union

import numpy as np
import torch
from torch import nn

__all__ = [
    'SpikingBatchNorm1d', 'SpikingEiNorm1d'
]

class SpikingBatchNorm1d(nn.Module):
    """Batch normalization wrapper."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        """Initialize module.

        Args:
            num_features: Number of feature channels.
            eps: Small constant for numerical stability.
            momentum: Momentum for running statistics.
            affine: Whether to include learnable affine parameters.
            track_running_stats: Whether to track running statistics.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = True

    def _set_visualize_cache(self, *args: torch.Tensor) -> None:
        """Cache tensors for visualization."""
        with torch.no_grad():
            inputs, output = args
            self.visualize_cache['param1:gamma'] = self.bn.weight.detach()
            self.visualize_cache['param2:beta'] = self.bn.bias.detach()
            self.visualize_cache['data1:input'] = inputs.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization.

        Args:
            inputs: Input tensor.

        Returns:
            Normalized output tensor.
        """
        output = self.bn.forward(inputs)
        if self._need_visualize:
            self._set_visualize_cache(inputs, output)
        return output

    def set_visualize(self, flag: bool) -> None:
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag


class SpikingEiNorm1d(nn.Module):
    """E-I layer for 1D inputs. This layer includes I-to-E path and 
    integrates inputs to E neurons. There is no explicit normalization
    (sub mean, div std) operation in this layer."""

    def __init__(self, num_features: int, prev_in_features: int, ei_ratio: int,
                 device: torch.device, output_layer: bool = False):
        """Initialize module.

        Args:
            num_features: Number of excitatory features.
            prev_in_features: Previous layer input dimension.
            ei_ratio: # E neurons / # I neurons.
            device: Device for parameter allocation.
            output_layer: Whether this is the final output layer.
        """
        super().__init__()
        self.n_e = num_features
        self.n_i = num_features // ei_ratio
        self.prev_in_features = prev_in_features
        self.device = device
        self.output_layer = output_layer

        self.weight_ei = nn.Parameter(
            torch.ones(self.n_e, self.n_i, device=self.device) / self.n_i,
            requires_grad=True,
        )
        self.gain_i = nn.Parameter(
            torch.empty(1, self.n_i, device=self.device), 
            requires_grad=True) # $g_I$ in paper
        self.gain_e = nn.Parameter(
            torch.ones(1, self.n_e, device=self.device), 
            requires_grad=True)   # $g_E$ in paper
        self.bias = nn.Parameter(
            torch.zeros(1, self.n_e, device=self.device), 
            requires_grad=True)  # $b_E$ in paper

        self.weight_ei.register_hook(lambda grad: grad / self.prev_in_features)

        self.visualize_cache = {}
        self._need_visualize = False
        self._inited = False

    def _dynamic_init(self, batch_stats: dict[str, Any]):
        """Initialize parameters based on input batch statistics."""
        with torch.no_grad():
            E_x = batch_stats['E_x']
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.gain_i.data = torch.ones(1, self.n_i, device=self.device) / \
                np.sqrt(self.prev_in_features) * np.sqrt(E_x_square + Var_x) / E_x  # Eq. 54 in paper

    def _clamp_parameters(self) -> None:
        """Clamp parameters to be non-negative."""
        with torch.no_grad():
            self.weight_ei.data.clamp_(min=0)
            self.gain_i.data.clamp_(min=0)
            self.gain_e.data.clamp_(min=0)

    def _set_visualize_cache(self, *args: torch.Tensor) -> None:
        """Cache tensors for visualization."""
        with torch.no_grad():
            I_ei, I_balanced, I_shunting, I_int = args
            self.visualize_cache['param1: weight_ei'] = self.weight_ei.detach()
            self.visualize_cache['param2: gain_i'] = self.gain_i.detach()
            self.visualize_cache['param3: gain_e'] = self.gain_e.detach()
            self.visualize_cache['param4: bias'] = self.bias.detach()
            self.visualize_cache['data1: I_ei'] = I_ei.detach()
            self.visualize_cache['data2: I_balanced'] = I_balanced.detach()
            self.visualize_cache['data3: I_shunting'] = I_shunting.detach()
            self.visualize_cache['data4: I_int'] = I_int.detach()

    def _replace_zero_with_second_min(self, inputs: torch.Tensor):
        """Replace zeros with the sample-wise second minimum using STE.

        Args:
            inputs: Input tensor that may contain zeros.

        Returns:
            Tensor where zeros are replaced to avoid division by zero.
        """
        has_zero = (inputs == 0.).any()

        if has_zero:
            # adaptive stabilization
            mask = inputs == 0.
            tmp = inputs.clone()
            tmp[mask] = float('inf')

            batch_size = tmp.size(0)
            tmp_flat = tmp.reshape(batch_size, -1)
            sample_wise_second_min_flat, _ = torch.min(tmp_flat, dim=1, keepdim=True)
            second_min_shape = [1] * (inputs.dim() - 1)
            second_min = sample_wise_second_min_flat.view(batch_size, *second_min_shape)
            # output = input + second_min * mask.float()
            forward_output = torch.where(mask, second_min, inputs)
            output = forward_output.detach() + (inputs - inputs.detach())  # STE
            return output
        return inputs

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, Union[dict, None]]):
        """Run a forward pass

        Args:
            inputs: Tuple of (I_ee, I_ie, batch_stats).

        Returns:
            Normalized current tensor, or balanced current if output layer.
        """
        self._clamp_parameters()
        I_ee, I_ie, batch_stats = inputs
        if not self._inited and batch_stats is not None:
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ei = torch.matmul(self.weight_ei, I_ie.T).T

        I_balanced = I_ee - I_ei

        I_shunting = torch.matmul(
            self.weight_ei,
            (self.gain_i * I_ie).T,
        ).T

        # Avoid division by zero in shunting term.
        I_shunting_adjusted = self._replace_zero_with_second_min(I_shunting)

        I_int = self.gain_e * I_balanced / I_shunting_adjusted + self.bias

        if self._need_visualize:
            self._set_visualize_cache(I_ei, I_balanced, I_shunting_adjusted, I_int)

        if self.output_layer:
            return I_balanced
        return I_int

    def set_visualize(self, flag: bool):
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag
