import logging
import os
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import average_repetition
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
from model_tools.activations.pca import LayerPCA
from result_caching import store_xarray, store


class ModelCommitment(BrainModel):
    def __init__(self, identifier, activations_model, layers):
        self.layers = layers
        self.region_assemblies = {}
        self.layer_model = LayerModel(identifier=identifier, activations_model=activations_model)
        # forward brain-interface methods
        self.look_at = self.layer_model.look_at
        self.start_task = self.layer_model.start_task

    def commit_region(self, region, assembly, assembly_stratification=None):
        self.region_assemblies[region] = (assembly, assembly_stratification)  # lazy, only run when actually needed

    def do_commit_region(self, region):
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.activations_model, layers=self.layers)
        assembly, assembly_stratification = self.region_assemblies[region]
        best_layer = layer_selection(assembly, assembly_stratification=assembly_stratification)
        self.layer_model.commit(region, best_layer)

    def start_recording(self, recording_target):
        if recording_target not in self.layer_model.region_layer_map:  # not yet committed
            self.do_commit_region(recording_target)
        return self.layer_model.start_recording(recording_target)

    @property
    def identifier(self):
        return self.layer_model.identifier


class PixelsToDegrees:
    CENTRAL_VISION_DEGREES = 10

    def __init__(self, target_pixels, target_degrees=CENTRAL_VISION_DEGREES):
        self.target_pixels = target_pixels
        self.target_degrees = target_degrees
        framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
        self._directory = os.path.join(framework_home, "stimuli-degrees")

    def __call__(self, stimuli):
        target_dir = os.path.join(self._directory, stimuli.name,
                                  f"target_{self.target_degrees}deg_{self.target_pixels}pix")
        os.makedirs(target_dir, exist_ok=True)
        image_paths = {image_id: self.convert_image(stimuli.get_image(image_id),
                                                    image_degrees=degrees, target_dir=target_dir)
                       for image_id, degrees in zip(stimuli['image_id'], stimuli['degrees'])}
        converted_stimuli = StimulusSet(stimuli)  # .copy() for some reason keeps the link to the old metadata
        converted_stimuli.name = f"{stimuli.name}-{self.target_degrees}degrees_{self.target_pixels}"
        converted_stimuli['degrees'] = self.target_degrees
        converted_stimuli.image_paths = image_paths
        converted_stimuli.original_paths = {converted_stimuli.image_paths[image_id]: stimuli.image_paths[image_id]
                                            for image_id in stimuli['image_id']}
        return converted_stimuli

    @classmethod
    def hook(cls, activations_extractor, target_pixels, target_degrees=CENTRAL_VISION_DEGREES):
        hook = PixelsToDegrees(target_pixels=target_pixels, target_degrees=target_degrees)
        handle = activations_extractor.register_stimulus_set_hook(hook)
        return handle

    def convert_image(self, image_path, image_degrees, target_dir):
        target_path = os.path.join(target_dir, os.path.basename(image_path))
        if not os.path.isfile(target_path):
            image = self._load_image(image_path)
            pixels_per_degree = self.target_pixels / self.target_degrees
            stimulus_pixels = self._round(image_degrees * pixels_per_degree)
            image = self._resize_image(image, image_size=stimulus_pixels)
            image = self._center_on_background(image, background_size=self.target_pixels)
            self._write(image, target_path=target_path)
            image.close()
        return target_path

    def _round(self, number):
        return np.array(number).round().astype(int)

    def _load_image(self, image_path):
        return Image.open(image_path)

    def _resize_image(self, image, image_size):
        return image.resize((image_size, image_size), Image.ANTIALIAS)

    def _center_on_background(self, center_image, background_size, background_color='gray'):
        image = Image.new('RGB', (background_size, background_size), background_color)
        center_topleft = self._round(np.subtract(background_size, center_image.size) / 2)
        image.paste(center_image, tuple(center_topleft))
        return image

    def _write(self, image, target_path):
        image.save(target_path)


class LayerModel(BrainModel):
    def __init__(self, identifier, activations_model, region_layer_map: Optional[dict] = None):
        self.identifier = identifier
        self.activations_model = activations_model
        self.region_layer_map = region_layer_map or {}
        self.recorded_regions = []

    def look_at(self, stimuli):
        layer_regions = {self.region_layer_map[region]: region for region in self.recorded_regions}
        assert len(layer_regions) == len(self.recorded_regions), f"duplicate layers for {self.recorded_regions}"
        activations = self.activations_model(stimuli, layers=list(layer_regions.keys()))
        activations['region'] = 'neuroid', [layer_regions[layer] for layer in activations['layer'].values]
        return activations

    def start_task(self, task):
        if task != BrainModel.Task.passive:
            raise NotImplementedError()

    def start_recording(self, recording_target):
        if str(recording_target) not in self.region_layer_map:
            raise NotImplementedError(f"Region {recording_target} is not committed")
        self.recorded_regions = [recording_target]

    def commit(self, region, layer):
        self.region_layer_map[region] = layer


class LayerSelection:
    def __init__(self, model_identifier, activations_model, layers):
        """
        :param model_identifier: this is separate from the container name because it only refers to
            the combination of (model, preprocessing), i.e. no mapping.
        """
        self.model_identifier = model_identifier
        self._layer_scoring = LayerScores(model_identifier=model_identifier, activations_model=activations_model)
        self.layers = layers
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly, assembly_stratification=None):
        # for layer-mapping, attach LayerPCA so that we can cache activations
        model_identifier = self.model_identifier
        pca_hooked = LayerPCA.is_hooked(self._layer_scoring._activations_model)
        if not pca_hooked:
            pca_handle = LayerPCA.hook(self._layer_scoring._activations_model, n_components=1000)
            identifier = self._layer_scoring._activations_model.identifier
            self._layer_scoring._activations_model.identifier = identifier + "-pca_1000"
            model_identifier += "-pca_1000"

        result = self._call(model_identifier=model_identifier, assembly_identifier=assembly.name,
                            assembly=assembly, assembly_stratification=assembly_stratification)

        if not pca_hooked:
            pca_handle.remove()
            self._layer_scoring._activations_model.identifier = identifier
        return result

    @store(identifier_ignore=['assembly'])
    def _call(self, model_identifier, assembly_identifier, assembly, assembly_stratification=None):
        benchmark = self._Benchmark(assembly, ceiler=InternalConsistency())
        self._logger.debug("Finding best layer")
        layer_scores = self._layer_scoring(benchmark=benchmark, layers=self.layers, prerun=True)

        self._logger.debug("Layer scores (unceiled): " + ", ".join([
            f"{layer} -> {layer_scores.raw.sel(layer=layer, aggregation='center').values:.3f}"
            f"+-{layer_scores.raw.sel(layer=layer, aggregation='error').values:.3f}"
            for layer in layer_scores['layer'].values]))
        best_layer = layer_scores['layer'].values[layer_scores.sel(aggregation='center').argmax()]
        return best_layer

    class _Benchmark(BenchmarkBase):
        def __init__(self, assembly_repetition, similarity_metric=None, ceiler=None):
            assert len(np.unique(assembly_repetition['region'])) == 1
            assert hasattr(assembly_repetition, 'repetition')
            self.region = np.unique(assembly_repetition['region'])[0]
            self.assembly = average_repetition(assembly_repetition)

            self._similarity_metric = similarity_metric or CrossRegressedCorrelation()
            identifier = f'{assembly_repetition.name}-layer_selection'
            ceiler = ceiler or InternalConsistency()
            super(LayerSelection._Benchmark, self).__init__(
                identifier=identifier, ceiling_func=lambda: ceiler(assembly_repetition))

        def __call__(self, candidate):
            candidate.start_recording(self.region)
            source_assembly = candidate.look_at(self.assembly.stimulus_set)
            raw_score = self._similarity_metric(source_assembly, self.assembly)
            return ceil_score(raw_score, self.ceiling)


class LayerScores:
    def __init__(self, model_identifier, activations_model):
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, benchmark, layers, benchmark_identifier=None, prerun=False):
        return self._call(model_identifier=self.model_identifier,
                          benchmark_identifier=benchmark_identifier or benchmark.identifier,
                          model=self._activations_model, benchmark=benchmark, layers=layers, prerun=prerun)

    @store_xarray(identifier_ignore=['model', 'benchmark', 'layers', 'prerun'], combine_fields={'layers': 'layer'})
    def _call(self, model_identifier, benchmark_identifier,  # storage fields
              model, benchmark, layers, prerun=False):
        if prerun:
            # pre-run activations together to avoid running every layer separately
            model(layers=layers, stimuli=benchmark.assembly.stimulus_set)

        layer_scores = []
        for layer in tqdm(layers, desc="layers"):
            layer_model = LayerModel(identifier=f"{model_identifier}-{layer}",  # per-layer identifier to avoid overlap
                                     activations_model=model, region_layer_map={benchmark.region: layer})
            score = benchmark(layer_model)
            score = score.expand_dims('layer')
            score['layer'] = [layer]
            layer_scores.append(score)
        layer_scores = Score.merge(*layer_scores)
        layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
        return layer_scores


def single_element(element_list):
    assert len(element_list) == 1
    return element_list[0]
