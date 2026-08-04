"""SEW-ResNet."""

import numpy as np
import torch
from torch import nn
from typing import Any

from modules.activation import LIF
from modules.blocks import (
    SpikingEiBasicBlock,
    SpikingEiBottleneck,
    SpikingStandardBasicBlock,
    SpikingStandardBottleneck,
)
from modules.conv2d import SpikingConv2d, SpikingEiConv2d
from modules.linear import SpikingEiLinear, SpikingLinear
from modules.norm1d import SpikingEiNorm1d
from modules.norm2d import SpikingBatchNorm2d, SpikingEiNorm2d
from utils.dim import AddTemporalDim, MergeTemporalDim, SplitTemporalDim

__all__ = [
    'SpikingResNet', 'SpikingEiResNet'
]

standard_layer_config = {
    18: {'block': SpikingStandardBasicBlock, 'layers': [2, 2, 2, 2]},
    34: {'block': SpikingStandardBasicBlock, 'layers': [3, 4, 6, 3]},
    50: {'block': SpikingStandardBottleneck, 'layers': [3, 4, 6, 3]},
    101: {'block': SpikingStandardBottleneck, 'layers': [3, 4, 23, 3]},
    152: {'block': SpikingStandardBottleneck, 'layers': [3, 8, 16, 3]}
}

ei_layer_config = {
    18: {'block': SpikingEiBasicBlock, 'layers': [2, 2, 2, 2]},
    34: {'block': SpikingEiBasicBlock, 'layers': [3, 4, 6, 3]},
    50: {'block': SpikingEiBottleneck, 'layers': [3, 4, 6, 3]},
    101: {'block': SpikingEiBottleneck, 'layers': [3, 4, 23, 3]},
    152: {'block': SpikingEiBottleneck, 'layers': [3, 8, 16, 3]}
}


class SpikingResNet(nn.Module):
    """SEW ResNet"""

    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                 neuron_config: dict, zero_init_residual: bool = False,
                 seq_input: bool = False):
        """Initialize model.

        Args:
            T: Number of time steps.
            num_layers: ResNet depth (e.g., 18, 34).
            in_channels: Number of input channels.
            n_outputs: Dimension of output.
            neuron_config: Spiking neuron configuration.
            zero_init_residual: Whether to zero-initialize residual branches.
            seq_input: Whether input is already sequential (T, B, ...).
        """
        super().__init__()
        self.in_channels = 64

        if not seq_input:
            self.init_expand = AddTemporalDim(T)
        else:
            self.init_expand = nn.Identity()
        self.init_merge = MergeTemporalDim(T)
        self.conv1 = SpikingConv2d(
            in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = SpikingBatchNorm2d(self.in_channels)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()

        layer_config = standard_layer_config[num_layers]
        block = layer_config['block']
        layers = layer_config['layers']

        self.layer1 = self._make_layers(
            block, neuron_config, T, 64, layers[0], stride=1
        )
        self.layer2 = self._make_layers(
            block, neuron_config, T, 128, layers[1], stride=2
        )
        self.layer3 = self._make_layers(
            block, neuron_config, T, 256, layers[2], stride=2
        )
        self.layer4 = self._make_layers(
            block, neuron_config, T, 512, layers[3], stride=2
        )

        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikingLinear(512 * block.expansion, n_outputs)
        self.final_split = SplitTemporalDim(T)

        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                nn.init.kaiming_normal_(
                    m.conv.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, SpikingBatchNorm2d):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SpikingStandardBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SpikingStandardBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layers(
        self,
        block: type[nn.Module],
        neuron_config: dict[str, Any],
        T: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a stage of residual blocks.

        Args:
            block: Residual block class.
            neuron_config: Spiking neuron configuration.
            T: Number of time steps.
            out_channels: Output channel count for the stage.
            num_blocks: Number of blocks in the stage.
            stride: Stride for the first block.

        Returns:
            nn.Sequential containing the stage.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                SpikingConv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                SpikingBatchNorm2d(out_channels * block.expansion),
                SplitTemporalDim(T),
                LIF(**neuron_config),
                MergeTemporalDim(T),
            )

        layers = []
        layers.append(
            block(T, self.in_channels, out_channels, neuron_config, stride, downsample)
        )

        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(T, self.in_channels, out_channels, neuron_config))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits averaged over time steps.
        """
        x = self.init_expand(x)
        x = self.init_merge(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.split1(x)
        x = self.lif1(x)
        x = self.merge1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.final_split(x)

        return x.mean(dim=0)

class SpikingEiResNet(nn.Module):
    """SEW ResNet with E-I layers."""

    def __init__(self, T: int, num_layers: int, in_channels: int, n_outputs: int,
                 neuron_config: dict, seq_input: bool, ei_ratio: int,
                 device: torch.device, rng: np.random.Generator):
        """Initialize a E-I SEW ResNet model.

        Args:
            T: Number of time steps.
            num_layers: ResNet depth (e.g., 18, 34).
            in_channels: Number of input channels.
            n_outputs: Dimension of output.
            neuron_config: Spiking neuron configuration.
            seq_input: Whether input is already sequential (T, B, ...).
            ei_ratio: # excitatory neurons / # inhibitory neurons.
            device: Device for parameter allocation.
            rng: Random generator for initialization.
        """
        super().__init__()
        self.in_channels = 64

        if not seq_input:
            self.init_expand = AddTemporalDim(T)
        else:
            self.init_expand = nn.Identity()
        self.init_merge = MergeTemporalDim(T)
        self.conv1 = SpikingEiConv2d(
            in_channels,
            self.in_channels,
            ei_ratio,
            device,
            rng,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # self.conv1 = SpikingEiConv2d(in_channels, self.in_channels, ei_ratio,
        #                              device, rng, kernel_size=7, stride=2,
        #                              padding=3, bias=False)
        self.norm1 = SpikingEiNorm2d(
            self.in_channels,
            in_channels * 3 * 3,
            ei_ratio,
            device=device,
        )
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)
        # self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()

        layer_config = ei_layer_config[num_layers]
        block = layer_config['block']
        layers = layer_config['layers']

        self.layer1 = self._make_layers(
            block, ei_ratio, neuron_config, T, 64, layers[0], rng, device, stride=1
        )
        self.layer2 = self._make_layers(
            block, ei_ratio, neuron_config, T, 128, layers[1], rng, device, stride=2
        )
        self.layer3 = self._make_layers(
            block, ei_ratio, neuron_config, T, 256, layers[2], rng, device, stride=2
        )
        self.layer4 = self._make_layers(
            block, ei_ratio, neuron_config, T, 512, layers[3], rng, device, stride=2
        )

        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikingEiLinear(512 * block.expansion, n_outputs, ei_ratio, device, rng)
        self.final_ei_norm = SpikingEiNorm1d(
            n_outputs,
            512 * block.expansion,
            ei_ratio,
            device,
            output_layer=True,
        )
        self.final_split = SplitTemporalDim(T)

    def _make_layers(
        self,
        block: type[nn.Module],
        ei_ratio: int,
        neuron_config: dict[str, Any],
        T: int,
        out_channels: int,
        num_blocks: int,
        rng: np.random.Generator,
        device: torch.device,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a stage of E-I residual blocks.

        Args:
            block: Residual block class.
            ei_ratio: # excitatory neurons / # inhibitory neurons.
            neuron_config: Spiking neuron configuration.
            T: Number of time steps.
            out_channels: Output channel count for the stage.
            num_blocks: Number of blocks in the stage.
            rng: Random generator for initialization.
            device: Device for parameter allocation.
            stride: Stride for the first block.

        Returns:
            nn.Sequential containing the stage.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                SpikingEiConv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    ei_ratio,
                    device,
                    rng,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                SpikingEiNorm2d(
                    out_channels * block.expansion,
                    self.in_channels * 1 * 1,
                    ei_ratio,
                    device=device,
                ),
                SplitTemporalDim(T),
                LIF(**neuron_config),
                MergeTemporalDim(T),
            )

        layers = []
        layers.append(
            block(
                T,
                self.in_channels,
                out_channels,
                neuron_config,
                ei_ratio,
                device,
                rng,
                stride,
                downsample,
            )
        )

        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(
                block(
                    T,
                    self.in_channels,
                    out_channels,
                    neuron_config,
                    ei_ratio,
                    device,
                    rng,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits averaged over time steps.
        """
        x = self.init_expand(x)
        x = self.init_merge(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.split1(x)
        x = self.lif1(x)
        x = self.merge1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.final_ei_norm(x)
        x = self.final_split(x)

        return x.mean(dim=0)
