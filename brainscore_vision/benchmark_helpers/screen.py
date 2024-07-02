"""
Methods to feed visual input to a system-under-test through the screen
"""
import copy
import logging
import os
import shutil
from typing import Union

import numpy as np
from PIL import Image
from pathlib import Path
from result_caching import store, is_iterable
from tqdm import tqdm

from brainio.stimuli import StimulusSet

framework_home = Path(os.getenv('BRAINSCORE_HOME', '~/.brain-score')).expanduser()
root_path = framework_home / "stimuli_on_screen"
_logger = logging.getLogger(__name__)


def place_on_screen(stimulus_set: StimulusSet,
                    target_visual_degrees: Union[int, float],
                    source_visual_degrees: Union[int, float, None] = None):
    _logger.debug(f"Converting {stimulus_set.identifier} to {target_visual_degrees} degrees")

    assert source_visual_degrees or 'degrees' in stimulus_set, \
        "Need to provide the source images' visual degrees either as a parameter or in the stimulus_set"
    assert not (source_visual_degrees and 'degrees' in stimulus_set), \
        "Got a parameter for the source images' visual degrees, but also found a 'degrees' column in the stimulus_set"
    inferred_visual_degrees = _determine_visual_degrees(source_visual_degrees, stimulus_set)
    if (inferred_visual_degrees == target_visual_degrees).all():
        return stimulus_set
    return _place_on_screen(stimuli_identifier=stimulus_set.identifier, stimulus_set=stimulus_set,
                            target_visual_degrees=target_visual_degrees, source_visual_degrees=source_visual_degrees)


def _determine_visual_degrees(visual_degrees, stimulus_set):
    if not visual_degrees:
        visual_degrees = stimulus_set['degrees']
    if not is_iterable(visual_degrees):
        visual_degrees = np.array([visual_degrees] * len(stimulus_set))
    return visual_degrees


@store(identifier_ignore=['stimulus_set'])
def _place_on_screen(stimuli_identifier: str, stimulus_set: StimulusSet,
                     target_visual_degrees: Union[int, float], source_visual_degrees: Union[int, float, None] = None):
    source_degrees_formatted = f"{source_visual_degrees}" if source_visual_degrees is None \
        else f"{source_visual_degrees:.2f}"  # make sure we do not try to print a None with 2 decimal places
    converted_stimuli_id = f"{stimuli_identifier}--target{target_visual_degrees:.2f}--source{source_degrees_formatted}"
    source_visual_degrees = _determine_visual_degrees(source_visual_degrees, stimulus_set)

    target_dir = root_path / converted_stimuli_id
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    image_converter = ImageConverter(target_dir=target_dir)

    converted_image_paths = {}
    for image_id, image_degrees in tqdm(zip(stimulus_set['stimulus_id'], source_visual_degrees),
                                        total=len(stimulus_set), desc='convert image degrees'):
        converted_image_path = image_converter.convert_image(image_path=stimulus_set.get_stimulus(image_id),
                                                             source_degrees=image_degrees,
                                                             target_degrees=target_visual_degrees)
        converted_image_paths[image_id] = converted_image_path
    converted_stimuli = StimulusSet(stimulus_set.copy(deep=True))  # without copy, it will link to the previous stim set
    converted_stimuli.stimulus_paths = converted_image_paths
    converted_stimuli.identifier = converted_stimuli_id
    converted_stimuli['degrees'] = target_visual_degrees
    converted_stimuli.original_paths = copy.deepcopy(stimulus_set.stimulus_paths)
    return converted_stimuli


class ImageConverter:
    def __init__(self, target_dir):
        self._target_dir = Path(target_dir)

    def convert_image(self, image_path, source_degrees: Union[int, float], target_degrees: Union[int, float]):
        if source_degrees == target_degrees:
            return image_path
        ratio = target_degrees / source_degrees
        with self._load_image(image_path) as image:
            converted_image = self.apply_ratio(image, ratio)
            target_path = str(self._target_dir / os.path.basename(image_path))
            self._write(converted_image, target_path=target_path)
            return target_path

    def apply_ratio(self, image: Image, ratio: float, background_color='gray'):
        image_size = np.array(image.size)
        target_image_size = (ratio * image_size).round().astype(int)
        if ratio >= 1:  # enlarge the image
            return self._enlarge(image, target_image_size, background_color=background_color)
        else:  # crop the image
            return self._center_crop(image, target_image_size)

    def _enlarge(self, image, target_size, background_color):
        background_image = Image.new('RGB', tuple(target_size), background_color)
        center_topleft = ((target_size - image.size) / 2).round().astype(int)
        background_image.paste(image, tuple(center_topleft))
        return background_image

    def _center_crop(self, image, crop_size):
        left, upper = ((image.size - crop_size) / 2).round().astype(int)
        right, lower = [left, upper] + crop_size
        image = image.crop((left, upper, right, lower))
        return image

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