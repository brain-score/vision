"""Multi-layer perceptron."""

from typing import Any

import torch
from torch import nn
import numpy as np

from modules.linear import SpikingLinear, SpikingEiLinear
from modules.norm1d import SpikingBatchNorm1d, SpikingEiNorm1d
from modules.activation import LIF
from utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim

__all__ = [
    'SpikingMLP', 'SpikingEiMLP'
]

layer_config = {
    2: [500],
    4: [500, 500, 300],
    6: [500, 500, 300, 300, 300],
    8: [500, 500, 300, 300, 300, 100, 100],
    12: [500, 500, 500, 500, 300, 300, 300, 300, 100, 100, 100],
    16: [500, 500, 500, 500, 500, 300, 300, 300, 300, 300, 100, 100, 100, 100, 100],
}


class SpikingMLP(nn.Module):
    """Standard MLP with spiking neurons."""

    def __init__(self, T: int, num_layers: int, n_inputs: int, n_outputs: int,
                 neuron_config: dict[str, Any], BN: bool):
        """Initialize model.

        Args:
            T: Total time steps.
            num_layers: Number of layers.
            n_inputs: Input feature dimension.
            n_outputs: Output feature dimension.
            neuron_config: Configuration of spiking neurons.
            BN: Whether to use batch normalization.
        """
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.BN = BN
        self.layers = self._build_model(layer_config[self.num_layers])

        self._need_visualize = False

    def _build_model(self, cfg: list[int]):
        """Build model according to the layer configuration.

        Args:
            cfg: Layer configuration list.

        Returns:
            Sequential model containing all layers.
        """
        layers = []
        in_features = self.n_inputs
        layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        for out_features in cfg:
            layers.append(SpikingLinear(in_features, out_features))
            if self.BN:
                layers.append(SpikingBatchNorm1d(out_features))
            layers.append(SplitTemporalDim(self.T))
            layers.append(LIF(**self.neuron_config))
            layers.append(MergeTemporalDim(self.T))
            in_features = out_features
        layers.append(SpikingLinear(cfg[-1], self.n_outputs))
        layers.append(SplitTemporalDim(self.T))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits averaged over time steps.
        """
        x = x.view(x.size(0), -1)
        if self._need_visualize:
            self._set_layer_visualize(True)
        output = self.layers(x)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(0)

    def set_visualize(self, flag: bool):
        """Enable or disable visualization mode.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag

    def get_visualize(self):
        """Return whether visualization is enabled."""
        return self._need_visualize

    def _set_layer_visualize(self, visualize: bool):
        """Set visualization flag for all layers.

        Args:
            visualize: Whether to enable visualization.
        """
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(visualize)


class SpikingEiMLP(nn.Module):
    """E-I MLP with spiking neurons."""

    def __init__(self, T: int, num_layers: int, n_inputs: int, n_outputs: int,
                 neuron_config: dict[str, Any], ei_ratio: int,
                 device: torch.device, rng: np.random.Generator):
        """Initialize model.

        Args:
            T: Total time steps.
            num_layers: Number of layers.
            n_inputs: Input feature dimension.
            n_outputs: Output feature dimension.
            neuron_config: Configuration of spiking neurons.
            ei_ratio: # excitatory neurons / # inhibitory neurons.
            device: Device for parameter allocation.
            rng: Random generator for initialization.
        """
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.ei_ratio = ei_ratio

        # used for dynamic initialization
        self.device = device
        self.rng = rng

        self.layers = self._build_model(layer_config[self.num_layers])
        self._need_visualize = False

    def _build_model(self, cfg: list[int]):
        """Build model according to the layer configuration.

        Args:
            cfg: Layer configuration list.

        Returns:
            Sequential model containing all layers.
        """
        layers = []
        in_features = self.n_inputs
        layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        for out_features in cfg:
            layers.append(
                SpikingEiLinear(
                    in_features,
                    out_features,
                    self.ei_ratio,
                    self.device,
                    self.rng,
                )
            )
            layers.append(
                SpikingEiNorm1d(
                    out_features,
                    in_features,
                    self.ei_ratio,
                    self.device,
                )
            )
            layers.append(SplitTemporalDim(self.T))
            layers.append(LIF(**self.neuron_config))
            layers.append(MergeTemporalDim(self.T))
            in_features = out_features
        layers.append(
            SpikingEiLinear(
                cfg[-1],
                self.n_outputs,
                self.ei_ratio,
                self.device,
                self.rng,
            )
        )
        layers.append(
            SpikingEiNorm1d(
                self.n_outputs,
                cfg[-1],
                self.ei_ratio,
                self.device,
                output_layer=True,
            )
        )
        layers.append(SplitTemporalDim(self.T))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output logits averaged over time steps.
        """
        x = x.view(x.size(0), -1)
        if self._need_visualize:
            self._set_layer_visualize(True)
        output = self.layers(x)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(0)

    def set_visualize(self, flag: bool) -> None:
        """Enable or disable visualization mode.

        Args:
            flag: Whether to enable visualization.
        """
        self._need_visualize = flag

    def get_visualize(self) -> bool:
        """Return whether visualization is enabled."""
        return self._need_visualize

    def _set_layer_visualize(self, flag: bool) -> None:
        """Set visualization flag for all layers.

        Args:
            flag: Whether to enable visualization.
        """
        for layer in self.layers:
            if hasattr(layer, 'set_visualize'):
                layer.set_visualize(flag)
