import functools
import importlib
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
from model_helpers.activations.core import ActivationsExtractorHelper
from model_helpers.activations.pytorch import PytorchWrapper
from brainio.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
import torch.hub
import ssl
import re
from brainscore_vision.model_helpers.s3 import load_weight_file
from torch.nn import Module
from collections import defaultdict


ssl._create_default_https_context = ssl._create_unverified_context


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


def get_model_list():
    return ['CORnet-S']


def get_model(name):
    assert name == 'CORnet-S'

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    mod = importlib.import_module(f'cornet.cornet_s')
    model_ctr = getattr(mod, 'CORnet_S')
    model = model_ctr()
    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="CORnet-S/cornet_s-1d3f7974.pth",
                                    version_id="NQqdW3MX1q1NgLp2suJDJS_iuwzWzDO.",
                                    sha1="a4bfd8eda33b45fd945da1b972ab0b7cad38d60f")
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)  # map onto cpu
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier="CORnet-S", model=model, preprocessing=preprocessing,
                                     separate_time=True)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'CORnet-S'
    return (['V1.output-t0'] +
               [f'{area}.output-t{timestep}'
                for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                for timestep in timesteps] +
               ['decoder.avgpool-t0']
            )


def get_bibtex(model_identifier):
    return """@inproceedings{KubiliusSchrimpf2019CORnet,
            abstract = {Deep convolutional artificial neural networks (ANNs) are the leading class of candidate models of the mechanisms of visual processing in the primate ventral stream. While initially inspired by brain anatomy, over the past years, these ANNs have evolved from a simple eight-layer architecture in AlexNet to extremely deep and branching architectures, demonstrating increasingly better object categorization performance, yet bringing into question how brain-like they still are. In particular, typical deep models from the machine learning community are often hard to map onto the brain's anatomy due to their vast number of layers and missing biologically-important connections, such as recurrence. Here we demonstrate that better anatomical alignment to the brain and high performance on machine learning as well as neuroscience measures do not have to be in contradiction. We developed CORnet-S, a shallow ANN with four anatomically mapped areas and recurrent connectivity, guided by Brain-Score, a new large-scale composite of neural and behavioral benchmarks for quantifying the functional fidelity of models of the primate ventral visual stream. Despite being significantly shallower than most models, CORnet-S is the top model on Brain-Score and outperforms similarly compact models on ImageNet. Moreover, our extensive analyses of CORnet-S circuitry variants reveal that recurrence is the main predictive factor of both Brain-Score and ImageNet top-1 performance. Finally, we report that the temporal evolution of the CORnet-S "IT" neural population resembles the actual monkey IT population dynamics. Taken together, these results establish CORnet-S, a compact, recurrent ANN, as the current best model of the primate ventral visual stream.},
            archivePrefix = {arXiv},
            arxivId = {1909.06161},
            author = {Kubilius, Jonas and Schrimpf, Martin and Hong, Ha and Majaj, Najib J. and Rajalingham, Rishi and Issa, Elias B. and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Schmidt, Kailyn and Nayebi, Aran and Bear, Daniel and Yamins, Daniel L. K. and DiCarlo, James J.},
            booktitle = {Neural Information Processing Systems (NeurIPS)},
            editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and D'Alch{\'{e}}-Buc, F. and Fox, E. and Garnett, R.},
            pages = {12785----12796},
            publisher = {Curran Associates, Inc.},
            title = {{Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs}},
            url = {http://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns},
            year = {2019}
            }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)