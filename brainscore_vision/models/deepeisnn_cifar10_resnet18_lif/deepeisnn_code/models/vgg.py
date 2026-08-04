"""VGG models."""

from typing import Any

import numpy as np
import torch
from torch import nn

from modules.activation import LIF
from modules.conv2d import SpikingConv2d, SpikingEiConv2d
from modules.linear import SpikingLinear, SpikingEiLinear
from modules.norm1d import SpikingBatchNorm1d, SpikingEiNorm1d
from modules.norm2d import SpikingBatchNorm2d, SpikingEiNorm2d
from utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim

__all__ = [
    'SpikingVGG', 'SpikingEiVGG'
]

layer_config = {
    8: [64, 'P', 128, 'P', 256, 'P', 512, 'P', 512, 'P'],
    11: [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    16: [
        64, 64, 'P',
        128, 128, 'P',
        256, 256, 256, 'P',
        512, 512, 512, 'P',
        512, 512, 512, 'P',
    ],
    19: [
        64, 64, 'P',
        128, 128, 'P',
        256, 256, 256, 256, 'P',
        512, 512, 512, 512, 'P',
        512, 512, 512, 512, 'P',
    ],
}

conv_config = {
    'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1,
    'groups': 1, 'padding_mode': 'zeros', 'bias': False
}

class SpikingVGG(nn.Module):
    """Standard VGG."""

    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                 neuron_config: dict[str, Any], light_classifier: bool,
                 dropout: float, seq_input: bool, BN: bool):
        """Initialize model.

        Args:
            T: Number of time steps.
            num_layers: VGG depth (e.g., 16, 19).
            in_channels: Number of input channels.
            n_outputs: Dimension of output.
            neuron_config: Spiking neuron configuration.
            light_classifier: Whether to use a lightweight classifier head.
            dropout: Dropout probability.
            seq_input: Whether input is already sequential (T, B, ...).
            BN: Whether to include batch normalization layers.
        """
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.BN = BN
        self.light_classifier = light_classifier
        self.dropout = dropout
        self.seq_input = seq_input
        self.conv_config = conv_config

        self.layers = self._build_model(layer_config[self.num_layers])
        self._need_visualize = False

    def _build_model(self, cfg: list[Any]) -> nn.Sequential:
        """Build the full VGG model.

        Args:
            cfg: Layer configuration list.

        Returns:
            Sequential model containing feature extractor and classifier.
        """
        layers = []
        if not self.seq_input:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._build_extractor(cfg)
        layers += self._build_classifier(cfg[-2])
        return nn.Sequential(*layers)

    def _build_extractor(self, cfg: list[Any]) -> list[nn.Module]:
        """Build the feature extractor portion.

        Args:
            cfg: Layer configuration list.

        Returns:
            List of feature extractor layers.
        """
        layers = []
        in_channels = self.in_channels
        for layer in cfg:
            if layer == 'P':
                layers.append(nn.AvgPool2d(2, 2, 0))
            else:
                layers.append(SpikingConv2d(in_channels, layer, **self.conv_config))
                if self.BN:
                    layers.append(SpikingBatchNorm2d(layer))
                layers.append(SplitTemporalDim(self.T))
                layers.append(LIF(**self.neuron_config))
                layers.append(MergeTemporalDim(self.T))
                in_channels = layer

        layers.append(nn.Flatten())
        return layers

    def _build_classifier(self, in_channels: int) -> list[nn.Module]:
        """Construct the classifier head.

        Args:
            in_channels: Input channels to the classifier.

        Returns:
            List of classifier layers.
        """
        layers = []
        if self.light_classifier:
            layers.append(SpikingLinear(in_channels * 1 * 1, self.n_outputs))
            layers.append(SplitTemporalDim(self.T))
            return layers

        layers.append(SpikingLinear(in_channels * 1 * 1, 4096))
        if self.BN:
            layers.append(SpikingBatchNorm1d(4096))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))

        layers.append(SpikingLinear(4096, 4096))
        if self.BN:
            layers.append(SpikingBatchNorm1d(4096))
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))
        layers.append(SpikingLinear(4096, self.n_outputs))
        layers.append(SplitTemporalDim(self.T))
        return layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Input tensor.

        Returns:
            Logits averaged over time steps.
        """
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.seq_input:
            inputs = inputs.transpose(0, 1)
        output = self.layers(inputs)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(dim=0)

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

class SpikingEiVGG(nn.Module):
    """VGG with E-I layers."""

    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                 neuron_config: dict[str, Any], light_classifier: bool,
                 dropout: float, seq_input: bool, ei_ratio: int,
                 device: torch.device, rng: np.random.Generator):
        """Initialize model.

        Args:
            T: Number of time steps.
            num_layers: VGG depth (e.g., 11, 16).
            in_channels: Number of input channels.
            n_outputs: Number of output classes.
            neuron_config: Spiking neuron configuration.
            light_classifier: Whether to use a lightweight classifier head.
            dropout: Dropout probability.
            seq_input: Whether input is already sequential (T, B, ...).
            ei_ratio: # excitatory neurons / # inhibitory neurons.
            device: Device for parameter allocation.
            rng: Random generator for initialization.
        """
        super().__init__()
        self.T = T
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.neuron_config = neuron_config
        self.light_classifier = light_classifier
        self.dropout = dropout
        self.seq_input = seq_input
        self.ei_ratio = ei_ratio
        self.conv_config = conv_config

        self.device = device
        self.rng = rng

        self.layers = self._build_model(layer_config[self.num_layers])
        self._need_visualize = False

    def _build_model(self, cfg: list[Any]) -> nn.Sequential:
        """Build the full VGG model.

        Args:
            cfg: Layer configuration list.

        Returns:
            Sequential model containing feature extractor and classifier.
        """
        layers = []
        if not self.seq_input:
            layers.append(AddTemporalDim(self.T))
        layers.append(MergeTemporalDim(self.T))
        layers += self._build_extractor(cfg)
        layers += self._build_classifier(cfg[-2])
        return nn.Sequential(*layers)

    def _build_extractor(self, cfg: list[Any]) -> list[nn.Module]:
        """Build the feature extractor portion.

        Args:
            cfg: Layer configuration list.

        Returns:
            List of feature extractor layers.
        """
        layers = []
        in_channels = self.in_channels
        for layer in cfg:
            if layer == 'P':
                layers.append(nn.AvgPool2d(2, 2, 0))
            else:
                layers.append(
                    SpikingEiConv2d(
                        in_channels,
                        layer,
                        self.ei_ratio,
                        self.device,
                        self.rng,
                        **self.conv_config,
                    )
                )
                layers.append(
                    SpikingEiNorm2d(
                        layer,
                        in_channels * self.conv_config['kernel_size'] ** 2,
                        self.ei_ratio,
                        device=self.device,
                    )
                )
                layers.append(SplitTemporalDim(self.T))
                layers.append(LIF(**self.neuron_config))
                layers.append(nn.Dropout(self.dropout))
                layers.append(MergeTemporalDim(self.T))
                in_channels = layer

        layers.append(nn.Flatten())
        return layers

    def _build_classifier(self, in_channels: int) -> list[nn.Module]:
        """Construct the classifier head.

        Args:
            in_channels: Input channels to the classifier.

        Returns:
            List of classifier layers.
        """
        layers = []
        if self.light_classifier:
            layers.append(
                SpikingEiLinear(
                    in_channels * 1 * 1,
                    self.n_outputs,
                    self.ei_ratio,
                    self.device,
                    self.rng,
                )
            )
            layers.append(
                SpikingEiNorm1d(
                    self.n_outputs,
                    in_channels * 1 * 1,
                    self.ei_ratio,
                    self.device,
                    output_layer=True,
                )
            )
            layers.append(SplitTemporalDim(self.T))
            return layers

        layers.append(
            SpikingEiLinear(
                in_channels * 1 * 1,
                4096,
                self.ei_ratio,
                self.device,
                self.rng,
            )
        )
        layers.append(
            SpikingEiNorm1d(
                4096,
                in_channels * 4 * 4,
                self.ei_ratio,
                self.device,
            )
        )
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))

        layers.append(
            SpikingEiLinear(
                4096,
                4096,
                self.ei_ratio,
                self.device,
                self.rng,
            )
        )
        layers.append(
            SpikingEiNorm1d(
                4096,
                4096,
                self.ei_ratio,
                self.device,
            )
        )
        layers.append(SplitTemporalDim(self.T))
        layers.append(LIF(**self.neuron_config))
        layers.append(nn.Dropout(self.dropout))
        layers.append(MergeTemporalDim(self.T))
        layers.append(
            SpikingEiLinear(
                4096,
                self.n_outputs,
                self.ei_ratio,
                self.device,
                self.rng,
            )
        )
        layers.append(
            SpikingEiNorm1d(
                self.n_outputs,
                4096,
                self.ei_ratio,
                self.device,
                output_layer=True,
            )
        )
        layers.append(SplitTemporalDim(self.T))
        return layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Input tensor.

        Returns:
            Logits averaged over time steps.
        """
        if self._need_visualize:
            self._set_layer_visualize(True)
        if self.seq_input:
            inputs = inputs.transpose(0, 1)
        output = self.layers(inputs)
        if self._need_visualize:
            self._set_layer_visualize(False)
        return output.mean(dim=0)

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
