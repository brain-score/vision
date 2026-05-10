"""Custom optimizers for models."""

from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer


class EiSGD(Optimizer):
    """SGD optimizer variant with optional E-I clamping."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        decay_mode: str = 'L2',
        nesterov: bool = False,
        clamped: bool = False,
    ) -> None:
        """Initialize the E-I-aware SGD optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            momentum: Momentum factor.
            dampening: Dampening for momentum.
            weight_decay: Weight decay coefficient.
            decay_mode: Weight decay mode ('L2' or 'L1').
            nesterov: Whether to use Nesterov momentum.
            clamped: Whether to clamp parameters to be non-negative.
        Raises:
            ValueError: If hyperparameters are out of range or invalid.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "decay_mode": decay_mode,
            "nesterov": nesterov,
            "clamped": clamped,
        }

        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss.

        Returns:
            Loss returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            decay_mode = group['decay_mode']
            nesterov = group['nesterov']
            clamped = group['clamped']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if weight_decay != 0:
                    if decay_mode == 'L2':
                        d_p = d_p.add(p, alpha=weight_decay)
                    elif decay_mode == 'L1':
                        d_p = d_p.add(torch.sign(p), alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(d_p).detach()

                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)
                if clamped:
                    p.data.clamp_(min=0.0)

        return loss
