import copy
import os
import cv2
import tempfile
from typing import Dict, Tuple, List, Union

import functools
import logging
from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm.auto import tqdm
import xarray as xr

from brainio.assemblies import NeuroidAssembly, walk_coords
from brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.utils import fullname
from result_caching import store_xarray


class Defaults:
    batch_size = 64


class ActivationsExtractorHelper:
    def __init__(self, get_activations, preprocessing, identifier=False, batch_size=Defaults.batch_size):
        """
        :param identifier: an activations identifier for the stored results file. False to disable saving.
        """
        self._logger = logging.getLogger(fullname(self))

        self._batch_size = batch_size
        self.identifier = identifier
        self.get_activations = get_activations
        self.preprocess = preprocessing or (lambda x: x)
        self._stimulus_set_hooks = {}
        self._batch_activations_hooks = {}
        self._microsaccade_helper = MicrosaccadeHelper()

    def __call__(self, stimuli, layers, stimuli_identifier=None, number_of_trials: int = 1,
                 require_variance: bool = False):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        :param number_of_trials: An integer that determines how many repetitions of the same model performs.
        :param require_variance: A bool that asks models to output different responses to the same stimuli (i.e.,
            allows stochastic responses to identical stimuli, even in otherwise deterministic base models). 
            We here implement this using microsaccades. For more, see ...

        """
        if require_variance:
            self._microsaccade_helper.number_of_trials = number_of_trials  # for use with microsaccades
        if (self._microsaccade_helper.visual_degrees is None) and require_variance:
            self._logger.debug("When using microsaccades for model commitments other than ModelCommitment, you should "
                               "set self.activations_model.set_visual_degrees(visual_degrees). Not doing so risks "
                               "breaking microsaccades.")
        if isinstance(stimuli, StimulusSet):
            function_call = functools.partial(self.from_stimulus_set, stimulus_set=stimuli)
        else:
            function_call = functools.partial(self.from_paths, stimuli_paths=stimuli)
        return function_call(
            layers=layers,
            stimuli_identifier=stimuli_identifier,
            require_variance=require_variance)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None, require_variance: bool = False):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file.
            False to disable saving. None to use `stimulus_set.identifier`
        """
        if stimuli_identifier is None and hasattr(stimulus_set, 'identifier'):
            stimuli_identifier = stimulus_set.identifier
        for hook in self._stimulus_set_hooks.copy().values():  # copy to avoid stale handles
            stimulus_set = hook(stimulus_set)
        stimuli_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
        activations = self.from_paths(stimuli_paths=stimuli_paths, layers=layers, stimuli_identifier=stimuli_identifier)
        activations = attach_stimulus_set_meta(activations,
                                               stimulus_set,
                                               number_of_trials=self._microsaccade_helper.number_of_trials,
                                               require_variance=require_variance)
        return activations

    def from_paths(self, stimuli_paths, layers, stimuli_identifier=None, require_variance=None):
        if layers is None:
            layers = ['logits']
        if self.identifier and stimuli_identifier:
            fnc = functools.partial(self._from_paths_stored,
                                    identifier=self.identifier,
                                    stimuli_identifier=stimuli_identifier,
                                    require_variance=require_variance)
        else:
            self._logger.debug(f"self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} "
                               f"are not set, will not store")
            fnc = self._from_paths
        if require_variance:
            activations = fnc(layers=layers, stimuli_paths=stimuli_paths, require_variance=require_variance)
        else:
            # When we are not asked for varying responses but receive `stimuli_paths` duplicates (e.g. multiple trials),
            # we first reduce them to only the paths that need to be run individually, compute activations for those,
            # and then expand the activations to all paths again. This is done here, before storing, so that we only
            # store the reduced activations.
            reduced_paths = self._reduce_paths(stimuli_paths)
            activations = fnc(layers=layers, stimuli_paths=reduced_paths, require_variance=require_variance)
            activations = self._expand_paths(activations, original_paths=stimuli_paths)
        return activations

    @store_xarray(identifier_ignore=['stimuli_paths', 'layers'], combine_fields={'layers': 'layer'})
    def _from_paths_stored(self, identifier, layers, stimuli_identifier,
                           stimuli_paths, number_of_trials: int = 1, require_variance: bool = False):
        return self._from_paths(layers=layers, stimuli_paths=stimuli_paths, require_variance=require_variance)

    def _from_paths(self, layers, stimuli_paths, require_variance: bool = False):
        if len(layers) == 0:
            raise ValueError("No layers passed to retrieve activations from")
        self._logger.info('Running stimuli')
        layer_activations = self._get_activations_batched(stimuli_paths, layers=layers, batch_size=self._batch_size,
                                                          require_variance=require_variance)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations=layer_activations, stimuli_paths=stimuli_paths, require_variance=require_variance)

    def _reduce_paths(self, stimuli_paths):
        return list(set(stimuli_paths))

    def _expand_paths(self, activations, original_paths):
        activations_paths = activations['stimulus_path'].values
        argsort_indices = np.argsort(activations_paths)
        sorted_x = activations_paths[argsort_indices]
        sorted_index = np.searchsorted(sorted_x, original_paths)
        index = [argsort_indices[i] for i in sorted_index]
        return activations[{'presentation': index}]

    def register_batch_activations_hook(self, hook):
        r"""
        The hook will be called every time a batch of activations is retrieved.
        The hook should have the following signature::

            hook(batch_activations) -> batch_activations

        The hook should return new batch_activations which will be used in place of the previous ones.
        """

        handle = HookHandle(self._batch_activations_hooks)
        self._batch_activations_hooks[handle.id] = hook
        return handle

    def register_stimulus_set_hook(self, hook):
        r"""
        The hook will be called every time before a stimulus set is processed.
        The hook should have the following signature::

            hook(stimulus_set) -> stimulus_set

        The hook should return a new stimulus_set which will be used in place of the previous one.
        """

        handle = HookHandle(self._stimulus_set_hooks)
        self._stimulus_set_hooks[handle.id] = hook
        return handle

    def _get_activations_batched(self, paths, layers, batch_size: int, require_variance: bool):
        layer_activations = OrderedDict()
        for batch_start in tqdm(range(0, len(paths), batch_size), unit_scale=batch_size, desc="activations"):
            batch_end = min(batch_start + batch_size, len(paths))
            batch_inputs = paths[batch_start:batch_end]

            batch_activations = OrderedDict()
            # compute activations on the entire batch one microsaccade shift at a time.
            for shift_number in range(self._microsaccade_helper.number_of_trials):
                activations = self._get_batch_activations(inputs=batch_inputs,
                                                          layer_names=layers,
                                                          batch_size=batch_size,
                                                          require_variance=require_variance,
                                                          trial_number=shift_number)

                for layer_name, layer_output in activations.items():
                    batch_activations.setdefault(layer_name, []).append(layer_output)

            # concatenate all microsaccade shifts in this batch (for example, if the model microsaccaded 15 times,
            #  the 15 microsaccaded layer_outputs are concatenated to the batch here.
            for layer_name, layer_outputs in batch_activations.items():
                batch_activations[layer_name] = np.concatenate(layer_outputs)

            for hook in self._batch_activations_hooks.copy().values():
                batch_activations = hook(batch_activations)

            # add this batch to layer_activations
            for layer_name, layer_output in batch_activations.items():
                layer_activations.setdefault(layer_name, []).append(layer_output)

        # concat all batches
        for layer_name, layer_outputs in layer_activations.items():
            layer_activations[layer_name] = np.concatenate(layer_outputs)

        return layer_activations  # this is all batches

    def _get_batch_activations(self, inputs, layer_names, batch_size: int, require_variance: bool = False,
                               trial_number: int = 1):
        inputs, num_padding = self._pad(inputs, batch_size)
        preprocessed_inputs = self.preprocess(inputs)
        preprocessed_inputs = self._microsaccade_helper.translate_images(images=preprocessed_inputs,
                                                                         image_paths=inputs,
                                                                         trial_number=trial_number,
                                                                         require_variance=require_variance)
        activations = self.get_activations(preprocessed_inputs, layer_names)
        assert isinstance(activations, OrderedDict)
        activations = self._unpad(activations, num_padding)
        if require_variance:
            self._microsaccade_helper.remove_temporary_files(preprocessed_inputs)
        return activations

    def set_visual_degrees(self, visual_degrees: float):
        """
        A method used by ModelCommitments to give the ActivationsExtractorHelper.MicrosaccadeHelper their visual
        degrees for performing microsaccades.
        """
        self._microsaccade_helper.visual_degrees = visual_degrees


    def _pad(self, batch_images, batch_size):
        num_images = len(batch_images)
        if num_images % batch_size == 0:
            return batch_images, 0
        num_padding = batch_size - (num_images % batch_size)
        padding = np.repeat(batch_images[-1:], repeats=num_padding, axis=0)
        return np.concatenate((batch_images, padding)), num_padding

    def _unpad(self, layer_activations, num_padding):
        return change_dict(layer_activations, lambda values: values[:-num_padding or None])

    def _package(self, layer_activations, stimuli_paths, require_variance: bool):
        shapes = [a.shape for a in layer_activations.values()]
        self._logger.debug(f"Activations shapes: {shapes}")
        self._logger.debug("Packaging individual layers")
        layer_assemblies = [self._package_layer(single_layer_activations,
                                                layer=layer,
                                                stimuli_paths=stimuli_paths,
                                                require_variance=require_variance) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        # complication: (non)neuroid_coords are taken from the structure of layer_assemblies[0] i.e. the 1st assembly;
        # using these names/keys for all assemblies results in KeyError if the first layer contains flatten_coord_names
        # (see _package_layer) not present in later layers, e.g. first layer = conv, later layer = transformer layer
        self._logger.debug(f"Merging {len(layer_assemblies)} layer assemblies")
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid'))
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = {coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
                          if set(dims) == {'neuroid'}}
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate((neuroid_coords[coord][1], layer_assembly[coord].values))
            assert layer_assemblies[0].dims == layer_assembly.dims
            for coord, dims, values in walk_coords(layer_assembly):
                if set(dims) == {'neuroid'}:
                    continue
                assert (values == nonneuroid_coords[coord][1]).all()

        neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
                          for coord, dims_values in neuroid_coords.items()}
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations: np.ndarray, layer: str, stimuli_paths: List[str], require_variance: bool = False):
        # activation shape is larger if variance in responses is required from the model by a factor of number_of_trials
        if require_variance:
            runs_per_image = self._microsaccade_helper.number_of_trials
        else:
            runs_per_image = 1
        assert layer_activations.shape[0] == len(stimuli_paths) * runs_per_image
        stimuli_paths = np.repeat(stimuli_paths, runs_per_image)
        activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
        flatten_coord_names = None
        if flatten_indices.shape[1] == 1:  # fully connected, e.g. classifier
            # see comment in _package for an explanation why we cannot simply have 'channel' for the FC layer
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 2:  # Transformer, e.g. ViT
            flatten_coord_names = ['channel', 'embedding']
        elif flatten_indices.shape[1] == 3:  # 2DConv, e.g. resnet
            flatten_coord_names = ['channel', 'channel_x', 'channel_y']
        elif flatten_indices.shape[1] == 4:  # temporal sliding window, e.g. omnivron
            flatten_coord_names = ['channel_temporal', 'channel_x', 'channel_y', 'channel']
        else:
            # we still package the activations, but are unable to provide channel information
            self._logger.debug(f"Unknown layer activations shape {layer_activations.shape}, not inferring channels")

        # build assembly
        coords = {'stimulus_path': ('presentation', stimuli_paths),
                  **self._microsaccade_helper.build_microsaccade_coords(stimuli_paths),
                  'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
                  'model': ('neuroid', [self.identifier] * activations.shape[1]),
                  'layer': ('neuroid', [layer] * activations.shape[1]),
                  }

        if flatten_coord_names:
            flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan
                                                       for sample_index in flatten_indices]
                              for i in range(len(flatten_coord_names))}
            coords = {**coords, **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}}
        layer_assembly = NeuroidAssembly(activations, coords=coords, dims=['presentation', 'neuroid'])
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
        return layer_assembly

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_paths = self.from_paths
        wrapper.register_batch_activations_hook = self.register_batch_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook


class MicrosaccadeHelper:
    """
    A class that allows ActivationsExtractorHelper to implement microsaccades.

    Human microsaccade amplitude varies by who you ask, an estimate might be <0.1 deg = 360 arcsec = 6arcmin.
    Our motivation to make use of such microsaccades is to obtain multiple different neural activities to the
    same input stimulus from non-stochastic models. This enables models to engage on e.g. psychophysical
    functions which often require variance for the same stimulus. In the current implementation,
    if `require_variance=True`, the model microsaccades in the preprocessed input space in sub-pixel increments,
    the extent and position of which are determined by `self._visual_degrees`, and
    `self.microsaccade_extent_degrees`.

    More information:
    --> Rolfs 2009 "Microsaccades: Small steps on a long way" Vision Research, Volume 49, Issue 20, 15
    October 2009, Pages 2415-2441.
    --> Haddad & Steinmann 1973 "The smallest voluntary saccade: Implications for fixation" Vision
    Research Volume 13, Issue 6, June 1973, Pages 1075-1086, IN5-IN6.
    Implemented by Ben Lonnqvist and Johannes Mehrer.
    """
    def __init__(self):
        self._logger = logging.getLogger(fullname(self))
        self.number_of_trials = 1  # for use with microsaccades.
        self.microsaccade_extent_degrees = 0.05  # how many degrees models microsaccade by default

        # a dict that contains two dicts, one for representing microsaccades in pixels, and one in degrees.
        #  Each dict inside contain image paths and their respective microsaccades. For example
        #  {'pixels': {'abc.jpg': [(0, 0), (1.5, 2)]}, 'degrees': {'abc.jpg': [(0., 0.), (0.0075, 0.001)]}}
        self.microsaccades = {'pixels': {}, 'degrees': {}}
        # Model visual degrees. Used for computing microsaccades in the space of degrees rather than pixels
        self.visual_degrees = None

    def translate_images(self, images: List[Union[str, np.ndarray]], image_paths: List[str], trial_number: int,
                         require_variance: bool) -> List[str]:
        """
        Translate images according to selected microsaccades, if microsaccades are required.

        :param images: A list of arrays.
        :param image_paths: A list of image paths. Both `image_paths` and `images` are needed since while both tf and
                             non-tf models preprocess images before this point, non-tf models' preprocessed images
                             are fixed as arrays when fed into here. As such, simply returning `image_paths` for
                             non-tf models would require double-loading of the images, which does not seem like a
                             good idea.
        """
        output_images = []
        for index, image_path in enumerate(image_paths):
            # When microsaccades are not used, skip computing them and return the base images.
            #  This iteration could be entirely skipped, but recording microsaccades for all images regardless
            #  of whether variance is required or not is convenient for adding an extra presentation dimension
            #  in the layer assembly later to keep track of as much metadata as possible, to avoid layer assembly
            #  collapse, or to avoid otherwise extraneous mock dims.
            #  The method could further be streamlined by calling `self.get_image_with_shape()` and
            #  `self.select_microsaccade` for all images regardless of require_variance, but it seems like a bad
            #  idea to introduce cv2 image loading for all models and images, regardless of whether they are actually
            #  microsaccading.
            if not require_variance:
                self.microsaccades['pixels'][image_path] = [(0., 0.)]
                self.microsaccades['degrees'][image_path] = [(0., 0.)]
                output_images.append(images[index])
            else:
                # translate images according to microsaccades if we are using microsaccades
                image, image_shape, image_is_channels_first = self.get_image_with_shape(images[index])
                microsaccade_location_pixels = self.select_microsaccade(image_path=image_path,
                                                                        trial_number=trial_number,
                                                                        image_shape=image_shape)
                return_string = True if isinstance(images[index], str) else False
                output_images.append(self.translate_image(image=image,
                                                          microsaccade_location=microsaccade_location_pixels,
                                                          image_shape=image_shape,
                                                          return_string=return_string,
                                                          image_is_channels_first=image_is_channels_first))
        return self.reshape_microsaccaded_images(output_images)

    def translate_image(self, image: str, microsaccade_location: Tuple[float, float], image_shape: Tuple[int, int],
                        return_string: bool, image_is_channels_first: bool) -> str:
        """Translates and saves a temporary image to temporary_fp."""
        translated_image = self.translate(image=image, shift=microsaccade_location, image_shape=image_shape,
                                          image_is_channels_first=image_is_channels_first)
        if not return_string:  # if the model accepts ndarrays after preprocessing, return one
            return translated_image
        else:  # if the model accepts strings after preprocessing, write temp file
            temp_file_descriptor, temporary_fp = tempfile.mkstemp(suffix=".png")
            os.close(temp_file_descriptor)
            if not cv2.imwrite(temporary_fp, translated_image):
                raise Exception(f"cv2.imwrite failed: {temporary_fp}")
        return temporary_fp

    def select_microsaccade(self, image_path: str, trial_number: int, image_shape: Tuple[int, int]
                            ) -> Tuple[float, float]:
        """
        A function for generating a microsaccade location. The function returns a tuple of pixel shifts expanding from
        the center of the image.

        Microsaccade locations are placed within a circle, evenly distributed across the entire area in a spiral,
        from the center to the circumference. We keep track of microsaccades both on a pixel and visual angle basis,
        but only pixel values are returned. This is because shifting the image using cv2 requires pixel representation.
        """
        # if we did not already compute `self.microsaccades`, we build them first.
        if image_path not in self.microsaccades.keys():
            self.build_microsaccades(image_path=image_path, image_shape=image_shape)
        return self.microsaccades['pixels'][image_path][trial_number]

    def build_microsaccades(self, image_path: str, image_shape: Tuple[int, int]):
        if image_shape[0] != image_shape[1]:
            self._logger.debug('Input image is not a square. Image dimension 0 is used to calculate the '
                               'extent of microsaccades.')

        assert self.visual_degrees is not None, (
            'self._visual_degrees is not set by the ModelCommitment, but microsaccades '
            'are in use. Set activations_model visual degrees in your commitment after defining '
            'your activations_model. For example, self.activations_model.set_visual_degrees'
            '(visual_degrees). For detailed information, see '
            ':meth:`~brainscore_vision.model_helpers.activations.ActivationsExtractorHelper.'
            '__call__`,')
        # compute the maximum radius of microsaccade extent in pixel space
        radius_ratio = self.microsaccade_extent_degrees / self.visual_degrees
        max_radius = radius_ratio * image_shape[0]  # maximum radius in pixels, set in self.microsaccade_extent_degrees

        selected_microsaccades = {'pixels': [], 'degrees': []}
        # microsaccades are placed in a spiral at sub-pixel increments
        a = max_radius / np.sqrt(self.number_of_trials)  # spiral coefficient to space microsaccades evenly
        for i in range(self.number_of_trials):
            r = np.sqrt(i / self.number_of_trials) * max_radius  # compute radial distance for the i-th point
            theta = a * np.sqrt(i) * 2 * np.pi / max_radius  # compute angle for the i-th point

            # convert polar coordinates to Cartesian, centered on the image
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            pixels_per_degree = self.calculate_pixels_per_degree_in_image(image_shape[0])
            selected_microsaccades['pixels'].append((x, y))
            selected_microsaccades['degrees'].append(self.convert_pixels_to_degrees((x, y), pixels_per_degree))

        # to keep consistent with number_of_trials, we count trial_number from 1 instead of from 0
        self.microsaccades['pixels'][image_path] = selected_microsaccades['pixels']
        self.microsaccades['degrees'][image_path] = selected_microsaccades['degrees']

    def unpack_microsaccade_coords(self, stimuli_paths: np.ndarray, pixels_or_degrees: str, dim: int):
        """Unpacks microsaccades from stimuli_paths into a single list to conform with coord requirements."""
        assert pixels_or_degrees == 'pixels' or pixels_or_degrees == 'degrees'
        unpacked_microsaccades = []
        for stimulus_path in stimuli_paths:
            for microsaccade in self.microsaccades[pixels_or_degrees][stimulus_path]:
                unpacked_microsaccades.append(microsaccade[dim])
        return unpacked_microsaccades

    def calculate_pixels_per_degree_in_image(self, image_width_pixels: int) -> float:
        """Calculates the pixels per degree in the image, assuming the calculation based on image width."""
        pixels_per_degree = image_width_pixels / self.visual_degrees
        return pixels_per_degree

    def build_microsaccade_coords(self, stimuli_paths: np.array) -> Dict:
        return {
            'microsaccade_shift_x_pixels': ('presentation', self.unpack_microsaccade_coords(
                np.unique(stimuli_paths),
                pixels_or_degrees='pixels',
                dim=0)),
            'microsaccade_shift_y_pixels': ('presentation', self.unpack_microsaccade_coords(
                np.unique(stimuli_paths),
                pixels_or_degrees='pixels',
                dim=1)),
            'microsaccade_shift_x_degrees': ('presentation', self.unpack_microsaccade_coords(
                np.unique(stimuli_paths),
                pixels_or_degrees='degrees',
                dim=0)),
            'microsaccade_shift_y_degrees': ('presentation', self.unpack_microsaccade_coords(
                np.unique(stimuli_paths),
                pixels_or_degrees='degrees',
                dim=1))
        }

    @staticmethod
    def convert_pixels_to_degrees(pixel_coords: Tuple[float, float], pixels_per_degree: float) -> Tuple[float, float]:
        degrees_x = pixel_coords[0] / pixels_per_degree
        degrees_y = pixel_coords[1] / pixels_per_degree
        return degrees_x, degrees_y

    @staticmethod
    def remove_temporary_files(temporary_file_paths: List[str]) -> None:
        """
        This function is used to manually remove all temporary file paths. We do this instead of using implicit
        python garbage collection to 1) ensure that tensorflow models have access to temporary files when needed;
        2) to make the point at which temporary files are removed explicit.
        """
        for temporary_file_path in temporary_file_paths:
            if isinstance(temporary_file_path, str):  # do not try to remove loaded images
                try:
                    os.remove(temporary_file_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def translate(image: np.array, shift: Tuple[float, float], image_shape: Tuple[int, int],
                  image_is_channels_first: bool) -> np.array:
        rows, cols = image_shape
        # translation matrix
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

        if image_is_channels_first:
            image = np.transpose(image, (1, 2, 0))  # cv2 expects channels last
        # Apply translation, filling new line(s) with line(s) closest to it(them).
        translated_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR,  # for sub-pixel shifts
                                          borderMode=cv2.BORDER_REPLICATE)
        if image_is_channels_first:
            translated_image = np.transpose(translated_image, (2, 0, 1))  # convert the image back to channels-first
        return translated_image

    @staticmethod
    def get_image_with_shape(image: Union[str, np.ndarray]) -> Tuple[np.array, Tuple[int, int], bool]:
        if isinstance(image, str):  # tf models return strings after preprocessing
            image = cv2.imread(image)
            rows, cols, _ = image.shape  # cv2 uses height, width, channels
            image_is_channels_first = False
        else:
            _, rows, cols, = image.shape  # pytorch uses channels, height, width
            image_is_channels_first = True
        return image, (rows, cols), image_is_channels_first

    @staticmethod
    def reshape_microsaccaded_images(images: List) -> Union[List[str], np.ndarray]:
        if any(isinstance(image, str) for image in images):
            return images
        return np.stack(images, axis=0)


def change_dict(d, change_function, keep_name=False, multithread=False):
    if not multithread:
        map_fnc = map
    else:
        pool = ThreadPool()
        map_fnc = pool.map

    def apply_change(layer_values):
        layer, values = layer_values
        values = change_function(values) if not keep_name else change_function(layer, values)
        return layer, values

    results = map_fnc(apply_change, d.items())
    results = OrderedDict(results)
    if multithread:
        pool.close()
    return results


def lstrip_local(path):
    parts = path.split(os.sep)
    try:
        start_index = parts.index('.brainio')
    except ValueError:  # not in list -- perhaps custom directory
        return path
    path = os.sep.join(parts[start_index:])
    return path


def attach_stimulus_set_meta(assembly, stimulus_set, number_of_trials: int, require_variance: bool = False):
    stimulus_paths = [str(stimulus_set.get_stimulus(stimulus_id)) for stimulus_id in stimulus_set['stimulus_id']]
    stimulus_paths = [lstrip_local(path) for path in stimulus_paths]
    assembly_paths = [lstrip_local(path) for path in assembly['stimulus_path'].values]

    # when microsaccades are used, we repeat stimulus_paths number_of_trials times to correctly populate the dim
    if require_variance:
        replication_factor = number_of_trials
    else:
        replication_factor = 1
    repeated_stimulus_paths = np.repeat(stimulus_paths, replication_factor)
    assert (np.array(assembly_paths) == np.array(repeated_stimulus_paths)).all()
    repeated_stimulus_ids = np.repeat(stimulus_set['stimulus_id'].values, replication_factor)

    if replication_factor > 1:
        # repeat over the presentation dimension to accommodate multiple runs per stimulus
        assembly = xr.concat([assembly for _ in range(replication_factor)], dim='presentation')
    assembly = assembly.reset_index('presentation')
    assembly['stimulus_path'] = ('presentation', repeated_stimulus_ids)
    assembly = assembly.rename({'stimulus_path': 'stimulus_id'})

    assert (np.array(assembly_paths) == np.array(stimulus_paths)).all()

    all_columns = []
    for column in stimulus_set.columns:
        repeated_values = np.repeat(stimulus_set[column].values, replication_factor)
        assembly = assembly.assign_coords({column: ('presentation', repeated_values)})  # assign multiple coords at once
        all_columns.append(column)

    presentation_coords = all_columns + [coord for coord, dims, values in walk_coords(assembly['presentation'])]
    assembly = assembly.set_index(presentation=list(set(presentation_coords)))  # assign MultiIndex
    return assembly


class HookHandle:
    next_id = 0

    def __init__(self, hook_dict):
        self.hook_dict = hook_dict
        self.id = HookHandle.next_id
        HookHandle.next_id += 1
        self._saved_hook = None

    def remove(self):
        hook = self.hook_dict[self.id]
        del self.hook_dict[self.id]
        return hook

    def disable(self):
        self._saved_hook = self.remove()

    def enable(self):
        self.hook_dict[self.id] = self._saved_hook
        self._saved_hook = None


def flatten(layer_output, return_index=False):
    flattened = layer_output.reshape(layer_output.shape[0], -1)
    if not return_index:
        return flattened

    def cartesian_product_broadcasted(*arrays):
        """
        http://stackoverflow.com/a/11146645/190597
        """
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        dtype = np.result_type(*arrays)
        rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in layer_output.shape[1:]])
    return flattened, index
