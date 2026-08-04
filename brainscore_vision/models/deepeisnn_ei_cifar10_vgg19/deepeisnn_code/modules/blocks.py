"""Building blocks for ResNet."""

import numpy as np
import torch
from torch import nn

from modules.activation import LIF
from modules.conv2d import SpikingConv2d, SpikingEiConv2d
from modules.norm2d import SpikingBatchNorm2d, SpikingEiNorm2d
from utils.dim import MergeTemporalDim, SplitTemporalDim

class SpikingStandardBasicBlock(nn.Module):
    """Standard ResNet basic block."""

    expansion = 1

    def __init__(self, T: int, in_channels: int, out_channels: int,
                 neuron_config: dict, stride: int = 1,
                 downsample: nn.Module = None):
        """Initialize a spiking basic block.

        Args:
            T: Number of time steps.
            in_channels: Input channel count.
            out_channels: Output channel count.
            neuron_config: Parameters for the spiking neuron.
            stride: Stride for the first convolution.
            downsample: Optional downsample module for the residual path.
        """
        super().__init__()

        self.conv1 = SpikingConv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.bn1 = SpikingBatchNorm2d(out_channels)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)

        self.conv2 = SpikingConv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn2 = SpikingBatchNorm2d(out_channels)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


class SpikingStandardBottleneck(nn.Module):
    """Standard ResNet bottleneck block."""

    expansion = 4

    def __init__(self, T: int, in_channels: int, out_channels: int,
                 neuron_config: dict, stride: int = 1,
                 downsample: nn.Module = None):
        """Initialize module.

        Args:
            T: Number of time steps.
            in_channels: Input channel count.
            out_channels: Output channel count.
            neuron_config: Parameters for the spiking neuron.
            stride: Stride for the second convolution.
            downsample: Optional downsample module for the residual path.
        """
        super().__init__()

        self.conv1 = SpikingConv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, bias=False)
        self.bn1 = SpikingBatchNorm2d(out_channels)
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)

        self.conv2 = SpikingConv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        self.bn2 = SpikingBatchNorm2d(out_channels)
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)

        self.conv3 = SpikingConv2d(out_channels, out_channels * self.expansion,
                                   kernel_size=1, stride=1, bias=False)
        self.bn3 = SpikingBatchNorm2d(out_channels * self.expansion)
        self.split3 = SplitTemporalDim(T)
        self.lif3 = LIF(**neuron_config)
        self.merge3 = MergeTemporalDim(T)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.split3(out)
        out = self.lif3(out)
        out = self.merge3(out)
        out += identity

        return out


class SpikingEiBasicBlock(nn.Module):
    """E-I ResNet basic block."""

    expansion = 1

    def __init__(self, T: int, in_channels: int, out_channels: int,
                 neuron_config: dict, ei_ratio: int, device: torch.device,
                 rng: np.random.Generator, stride: int = 1,
                 downsample: nn.Module = None):
        """Initialize module.

        Args:
            T: Number of time steps.
            in_channels: Input channel count.
            out_channels: Output channel count.
            neuron_config: Parameters for the spiking neuron.
            ei_ratio: # E neurons / # I neurons.
            device: Device for parameter initialization.
            rng: Random generator for initialization.
            stride: Stride for the first convolution.
            downsample: Optional downsample module for the residual path.
        """
        super().__init__()

        self.conv1 = SpikingEiConv2d(
            in_channels,
            out_channels,
            ei_ratio,
            device,
            rng,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = SpikingEiNorm2d(
            out_channels,
            in_channels * 3 * 3,
            ei_ratio,
            device=device,
        )
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)

        self.conv2 = SpikingEiConv2d(
            out_channels,
            out_channels,
            ei_ratio,
            device,
            rng,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = SpikingEiNorm2d(
            out_channels,
            out_channels * 3 * 3,
            ei_ratio,
            device=device,
        )
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition.
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class SpikingEiBottleneck(nn.Module):
    """E-I ResNet bottleneck block."""

    expansion = 4

    def __init__(self, T: int, in_channels: int, out_channels: int,
                 neuron_config: dict, ei_ratio: int, device: torch.device,
                 rng: np.random.Generator, stride: int = 1,
                 downsample: nn.Module = None):
        """Initialize module.

        Args:
            T: Number of time steps.
            in_channels: Input channel count.
            out_channels: Output channel count.
            neuron_config: Parameters for the spiking neuron.
            ei_ratio: # E neurons / # I neurons.
            device: Device for parameter initialization.
            rng: Random generator for initialization.
            stride: Stride for the second convolution.
            downsample: Optional downsample module for the residual path.
        """
        super().__init__()

        self.conv1 = SpikingEiConv2d(
            in_channels,
            out_channels,
            ei_ratio,
            device,
            rng,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm1 = SpikingEiNorm2d(
            out_channels,
            in_channels * 1 * 1,
            ei_ratio,
            device=device,
        )
        self.split1 = SplitTemporalDim(T)
        self.lif1 = LIF(**neuron_config)
        self.merge1 = MergeTemporalDim(T)

        self.conv2 = SpikingEiConv2d(
            out_channels,
            out_channels,
            ei_ratio,
            device,
            rng,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm2 = SpikingEiNorm2d(
            out_channels,
            out_channels * 3 * 3,
            ei_ratio,
            device=device,
        )
        self.split2 = SplitTemporalDim(T)
        self.lif2 = LIF(**neuron_config)
        self.merge2 = MergeTemporalDim(T)

        self.conv3 = SpikingEiConv2d(
            out_channels,
            out_channels * self.expansion,
            ei_ratio,
            device,
            rng,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.norm3 = SpikingEiNorm2d(
            out_channels * self.expansion,
            out_channels * self.expansion * 1 * 1,
            ei_ratio,
            device=device,
        )
        self.split3 = SplitTemporalDim(T)
        self.lif3 = LIF(**neuron_config)
        self.merge3 = MergeTemporalDim(T)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition.
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.split1(out)
        out = self.lif1(out)
        out = self.merge1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.split2(out)
        out = self.lif2(out)
        out = self.merge2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.split3(out)
        out = self.lif3(out)
        out = self.merge3(out)
        out += identity

        return out
