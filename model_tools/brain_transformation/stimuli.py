import os

import numpy as np
from PIL import Image

from brainio_base.stimuli import StimulusSet


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
