"""Spiking linear layers, including E-I variants."""

import numpy as np
import torch
from torch import nn

__all__ = [
    'SpikingLinear', 'SpikingEiLinear'
]


class SpikingLinear(nn.Module):
    """Linear layer wrapper."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize module.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = True

    def _set_visualize_cache(self, *args: torch.Tensor) -> None:
        """Cache tensors for visualization."""
        with torch.no_grad():
            inputs, output = args
            self.visualize_cache['param1:weight'] = self.linear.weight.detach()
            bias = self.linear.bias.detach() if self.linear.bias is not None else None
            if bias is not None:
                self.visualize_cache['param2:bias'] = bias

            self.visualize_cache['data1:input'] = inputs.detach()
            self.visualize_cache['data2:output'] = output.detach()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation.

        Args:
            inputs: Input tensor.

        Returns:
            Transformed output tensor.
        """
        output = self.linear.forward(inputs)
        if self._need_visualize:
            self._set_visualize_cache(inputs, output)
        return output

    def set_visualize(self, flag: bool):
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag


class SpikingEiLinear(nn.Module):
    """E-I linear layer."""

    def __init__(self, in_features: int, out_features: int, ei_ratio: int,
                 device: torch.device, rng: np.random.Generator,
                 output_layer: bool = False):
        """Initialize module.

        Args:
            in_features: Number of input features.
            out_features: Number of excitatory output features.
            ei_ratio: # E neurons / # I neurons.
            device: Device for parameter allocation.
            rng: Random generator for initialization.
            output_layer: Whether this is the final output layer.
        """

        super().__init__()
        self.in_features = in_features
        self.n_e = out_features
        self.n_i = self.n_e // ei_ratio
        self.device = device
        self.rng = rng
        self.output_layer = output_layer
        self.weight_ee = nn.Parameter(
            torch.empty(self.n_e, self.in_features, device=self.device),
            requires_grad=True,
        )
        self.weight_ie = nn.Parameter(
            torch.empty(self.n_i, self.in_features, device=self.device),
            requires_grad=True,
        )
        self.exp_scale: float | None = None

        self._need_visualize = False
        self.visualize_cache = {}
        self._inited = False

    def _clamp_parameters(self):
        """Clamp parameters to be non-negative."""
        with torch.no_grad():
            self.weight_ee.clamp_(min=0)
            self.weight_ie.clamp_(min=0)

    def _dynamic_init(self, batch_stats: dict[str, float]):
        """Initialize weights based on input batch statistics."""
        with torch.no_grad():
            Var_x = batch_stats['Var_x']
            E_x_square = batch_stats['E_x_square']
            self.exp_scale = np.sqrt(Var_x / (self.in_features * (E_x_square + Var_x)))

            weight_ee_np = self.rng.exponential(
                scale=self.exp_scale,
                size=(self.n_e, self.in_features),
            )
            self.weight_ee.data = torch.from_numpy(weight_ee_np).float().to(self.device)

            weight_ie_np = self.rng.exponential(
                scale=self.exp_scale,
                size=(self.n_i, self.in_features),
            )
            self.weight_ie.data = torch.from_numpy(weight_ie_np).float().to(self.device)

    def _get_batch_stats(self, x: torch.Tensor) -> dict[str, float]:
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

    def _set_visualize_cache(self, *args: torch.Tensor):
        """Cache tensors for visualization."""
        with torch.no_grad():
            inputs, I_ee, I_ie = args
            self.visualize_cache['param1:weight_ee'] = self.weight_ee.detach()
            self.visualize_cache['param2:weight_ie'] = self.weight_ie.detach()
            I_ie_np = I_ie.detach() if not self.output_layer else None
            self.visualize_cache['data1:input'] = inputs.detach()
            self.visualize_cache['data2:I_ee'] = I_ee.detach()
            if I_ie is not None:
                self.visualize_cache['data3:I_ie'] = I_ie_np

    def forward(self, x: torch.Tensor):
        """Apply the E-I linear transformation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor or tuple of currents and stats depending on layer type.
        """
        self._clamp_parameters()
        batch_stats = None
        if not self._inited:
            batch_stats = self._get_batch_stats(x)
            self._dynamic_init(batch_stats)
            self._inited = True

        I_ee = torch.matmul(self.weight_ee, x.T).T
        I_ie = torch.matmul(self.weight_ie, x.T).T
        if self._need_visualize:
            self._set_visualize_cache(x, I_ee, I_ie)
        if self.output_layer:
            return I_ee
        return I_ee, I_ie, batch_stats

    def set_visualize(self, flag: bool):
        """Enable or disable visualization cache.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag
