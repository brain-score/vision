from ._builder import *
from ._helpers import *
from ._manipulate import *
from ._prune import *

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", DeprecationWarning)

def overlay_external_default_cfg(default_cfg, kwargs):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg.
    """
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)
