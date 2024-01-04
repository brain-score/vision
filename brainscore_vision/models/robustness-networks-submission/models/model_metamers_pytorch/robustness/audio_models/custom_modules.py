import torch as ch
import numpy as np
from torch import nn
from robustness.audio_functions import audio_transforms
from torch._jit_internal import _copy_to_script_wrapper
import numpy as np
from .layers.padding import pad_valid_time, pad_same

class FakeReLU(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)

class SequentialWithArgs(ch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class AudioInputRepresentation(ch.nn.Module):
    '''
    A module (custom layer) for turning the audio signal into a
    representation for training, ie using a mel spectrogram or a
    cochleagram.
    '''
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioInputRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Functions for the representations are defined in the audio_transforms
        # library, but we only use the foreground audio here.
        self.full_rep = audio_transforms.AudioToAudioRepresentation(rep_type,
                                                                    rep_kwargs,
                                                                    compression_type,
                                                                    compression_kwargs)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        # print(self.full_rep)
        x, _ = self.full_rep(x, None)
        return x

class SequentialAttacker(ch.nn.Module):
    r"""A sequential container with additional kwargs for attacker models.
    Based on ch.nn.Sequential
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(SequentialAttacker, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(SequentialAttacker, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

