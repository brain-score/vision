""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from .clip_grad import dispatch_clip_grad


class ApexScaler:
    state_dict_key = "amp"

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
            optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
            
class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            norm = torch.tensor(0.)
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            #norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
