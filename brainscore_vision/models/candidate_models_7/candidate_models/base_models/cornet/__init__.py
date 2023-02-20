import importlib
import logging
import os
from collections import defaultdict

import functools
import re
import torch
from torch.nn import Module

from brainio_base.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from candidate_models import s3
from candidate_models.base_models.cornet.cornet_r2 import fix_state_dict_naming as fix_r2_state_dict_naming
from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.activations.pytorch import PytorchWrapper

_logger = logging.getLogger(__name__)


def cornet(identifier, separate_time=True):
    cornet_type = re.match('CORnet-(.*)', identifier).group(1)
    if cornet_type.lower() == 'r2':
        from .cornet_r2 import CORNetR2
        model_ctr = CORNetR2
    elif cornet_type.lower() == 's10':
        from .cornet_s_10 import CORnet_S as CORnet_S10
        model_ctr = CORnet_S10
    elif cornet_type.lower() == 's222':
        from .cornet_s_222 import CORnet_S as CORnet_S222
        model_ctr = CORnet_S222
    elif cornet_type.lower() == 's444':
        from .cornet_s_444 import CORnet_S as CORnet_S444
        model_ctr = CORnet_S444
    elif cornet_type.lower() == 's484':
        from .cornet_s_484 import CORnet_S as CORnet_S484
        model_ctr = CORnet_S484
    else:
        mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
        model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
    model = model_ctr()

    WEIGHT_MAPPING = {
        'Z': 'cornet_z-5c427c9c.pth',
        'R': 'cornet_r_epoch25.pth.tar',
        'S': 'cornet_s_epoch43.pth.tar',
        'R2': 'cornet_r2_epoch_60.pth.tar',
        'S10': 'cornet_s10_latest.pth.tar',
        'S222': 'cornet_s222_latest.pth.tar',
        'S444': 'cornet_s444_latest.pth.tar',
        'S484': 'cornet_s484_latest.pth.tar',
    }

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
    weightsdir_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'cornet'))
    weights_path = os.path.join(weightsdir_path, WEIGHT_MAPPING[cornet_type.upper()])
    if not os.path.isfile(weights_path):
        _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
        os.makedirs(weightsdir_path, exist_ok=True)
        s3.download_file(WEIGHT_MAPPING[cornet_type.upper()], weights_path, bucket='cornet-models')
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)  # map onto cpu
    if cornet_type.lower() == 'r2':
        checkpoint['state_dict'] = fix_r2_state_dict_naming(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap

    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing,
                                     separate_time=separate_time)
    wrapper.image_size = 224
    return wrapper


class TemporalPytorchWrapper(PytorchWrapper):
    def __init__(self, *args, separate_time=True, **kwargs):
        self._separate_time = separate_time
        super(TemporalPytorchWrapper, self).__init__(*args, **kwargs)

    def _build_extractor(self, *args, **kwargs):
        if self._separate_time:
            return TemporalExtractor(*args, **kwargs)
        else:
            return super(TemporalPytorchWrapper, self)._build_extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        # reset
        self._layer_counter = defaultdict(lambda: 0)
        self._layer_hooks = {}
        return super(TemporalPytorchWrapper, self).get_activations(images=images, layer_names=layer_names)

    def register_hook(self, layer, layer_name, target_dict):
        layer_name = self._strip_layer_timestep(layer_name)
        if layer_name in self._layer_hooks:  # add hook only once for multiple timesteps
            return self._layer_hooks[layer_name]

        def hook_function(_layer, _input, output):
            target_dict[f"{layer_name}-t{self._layer_counter[layer_name]}"] = PytorchWrapper._tensor_to_numpy(output)
            self._layer_counter[layer_name] += 1

        hook = layer.register_forward_hook(hook_function)
        self._layer_hooks[layer_name] = hook
        return hook

    def get_layer(self, layer_name):
        layer_name = self._strip_layer_timestep(layer_name)
        return super(TemporalPytorchWrapper, self).get_layer(layer_name)

    def _strip_layer_timestep(self, layer_name):
        match = re.search('-t[0-9]+$', layer_name)
        if match:
            layer_name = layer_name[:match.start()]
        return layer_name


class TemporalExtractor(ActivationsExtractorHelper):
    # `from_paths` is the earliest method at which we can interject because calls below are stored and checked for the
    # presence of all layers which, for CORnet, are passed as e.g. `IT.output-t0`.
    # This code re-arranges the time component.
    def from_paths(self, *args, **kwargs):
        raw_activations = super(TemporalExtractor, self).from_paths(*args, **kwargs)
        # introduce time dimension
        regions = defaultdict(list)
        for layer in set(raw_activations['layer'].values):
            match = re.match(r'(([^-]*)\..*|logits|avgpool)-t([0-9]+)', layer)
            region, timestep = match.group(2) if match.group(2) else match.group(1), match.group(3)
            stripped_layer = match.group(1)
            regions[region].append((layer, stripped_layer, timestep))
        activations = {}
        for region, time_layers in regions.items():
            for (full_layer, stripped_layer, timestep) in time_layers:
                region_time_activations = raw_activations.sel(layer=full_layer)
                region_time_activations['layer'] = 'neuroid', [stripped_layer] * len(region_time_activations['neuroid'])
                activations[(region, timestep)] = region_time_activations
        for key, key_activations in activations.items():
            region, timestep = key
            key_activations['region'] = 'neuroid', [region] * len(key_activations['neuroid'])
            activations[key] = NeuroidAssembly([key_activations.values], coords={
                **{coord: (dims, values) for coord, dims, values in walk_coords(activations[key])
                   if coord != 'neuroid_id'},  # otherwise, neuroid dim will be as large as before with nans
                **{'time_step': [int(timestep)]}
            }, dims=['time_step'] + list(key_activations.dims))
        activations = list(activations.values())
        activations = merge_data_arrays(activations)
        # rebuild neuroid_id without timestep
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            activations[coord].values for coord in ['model', 'region', 'neuroid_num']])]
        activations['neuroid_id'] = 'neuroid', neuroid_id
        return activations
