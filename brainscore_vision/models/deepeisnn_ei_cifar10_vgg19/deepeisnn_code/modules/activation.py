"""Customed activations."""

from typing import Any, Union

import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.neuron import LIFNode

surrogate_dict = {
    'sigmoid': surrogate.Sigmoid,
    'atan': surrogate.ATan,
    'leaky_relu': surrogate.LeakyKReLU,
}

class LIF(LIFNode):
    """LIF neuron wrapper with named surrogate functions."""

    def __init__(self, tau: float = 2.0, surrogate_function: str = 'sigmoid',
                 step_mode: str = 'm', decay_input: bool = False,
                 v_threshold: float = 1.0, v_reset: float = None,
                 detach_reset: bool = True):
        """Initialize the LIF neuron with a chosen surrogate function.

        Args:
            tau: Membrane time constant.
            surrogate_function: Name of surrogate function to use.
            step_mode: Step mode for the neuron.
            decay_input: Whether to decay input current.
            v_threshold: Spike threshold voltage.
            v_reset: Reset voltage after a spike.
            detach_reset: Whether to detach reset from autograd.
        """

        super().__init__(float(tau), decay_input, v_threshold, v_reset,
                         surrogate_function=surrogate_dict[surrogate_function](),
                         detach_reset=detach_reset, step_mode=step_mode)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, None]:
        """Run a forward pass.

        Args:
            x: Input tensor.
            *args: Additional positional arguments forwarded to parent.
            **kwargs: Additional keyword arguments forwarded to parent.

        Returns:
            Spike tensor.
        """
        spike = super().forward(x, *args, **kwargs)
        return spike
