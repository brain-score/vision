import logging
import xarray as xr
from ..spec import Spec
from ..inputs.base import Stimulus

from typing import List


logger = logging.getLogger(__name__)


PARAM_SYMBOL = '@'


class ActivationWrapper:
    identifier = None
    spec = Spec()
    extractor_cls=None

    def __init__(
            self,
            before_get_activations_hooks=None,
            after_get_activations_hooks=None,
            before_call_hooks=None,
            after_call_hooks=None,
        ):
        self.before_get_activations_hooks = before_get_activations_hooks or []
        self.after_get_activations_hooks = after_get_activations_hooks or []
        self.before_call_hooks = before_call_hooks or []
        self.after_call_hooks = after_call_hooks or []
        self._extractor = self._build_extractor(
                identifier=self.identifier if self.identifier is not None 
                else self.__class__.__name__, wrapper=self)
        self._extractor.insert_attrs(self)
        self._register_spec_hooks()

    # List[Stimulus] -> xr.DataArray
    def get_activations(self, inputs: List[Stimulus], layers):
        return xr.DataArray()
    
    # Register a hook (inputs, layers) -> (inputs, layers)
    def register_before_get_activations_hook(self, hook, location=0):
        self.before_get_activations_hooks.insert(location, hook)
    
    # Register a hook (inputs, layers, layer_results) -> (inputs, layers, layer_results)
    def register_after_get_activations_hook(self, hook, location=-1):
        self.after_get_activations_hooks.insert(location, hook)

    # Register a hook (inputs) -> (inputs)
    def register_before_call_hook(self, hook, location=0):
        self.before_call_hooks.insert(location, hook)

    # Register a hook (inputs, outputs) -> (inputs, outputs)
    def register_after_call_hook(self, hook, location=-1):
        self.after_call_hooks.insert(location, hook)
    
    def _register_spec_hooks(self):
        before_hooks, after_hooks = make_spec_hooks(self.spec)
        # spec hooks are in the most inner layer
        self.register_before_get_activations_hook(before_hooks, location=-1)
        self.register_after_get_activations_hook(after_hooks, location=0)
    
    @classmethod
    def bind_extractor_cls(cls, extractor_cls):
        cls.extractor_cls = extractor_cls

    def _build_extractor(self, identifier, get_activations, *args, **kwargs):
        if self.extractor_cls is None:
            raise ValueError("Extractor class is not set. Please call bind_extractor before using the model.")
        return self.extractor_cls(
            identifier=identifier, get_activations=get_activations, *args, **kwargs)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)
    
    @property
    def layers(self):
        return list(self.spec['activation'].keys())
            
def make_spec_hooks(spec):
    return