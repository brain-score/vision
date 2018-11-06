from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import peewee
from PIL import Image

from brainscore.lookup import pwdb


class StimulusSet(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ["image_paths", "get_image"]
    _internal_names_set = set(_internal_names)

    CENTRAL_VISION_DEGREES = 10

    def get_image(self, image_id, pixels_central_vision=None, output_directory=None):
        """
        :param image_id:
        :param pixels_central_vision: (width, height) of how many pixels are equivalent to central vision
                (10 degree visual angle). The image will be resized and centrally placed on gray background accordingly.
        :param output_directory: where to write the image when resizing it according to `pixels_central_vision`.
                Temporary directory by default
        :return: the path to the image.
        """
        image_path = self.image_paths[image_id]
        if not pixels_central_vision:
            return image_path
        if not output_directory:
            tempdir = tempfile.TemporaryDirectory()  # keep this object in context to avoid cleanup
            output_directory = tempdir.name
        image = self._load_image(image_path)
        degrees_ratio = self.degrees[self.image_id == image_id] / self.CENTRAL_VISION_DEGREES
        stimuli_size = (pixels_central_vision * degrees_ratio.values).round().astype(int)
        image = self._resize_image(image, image_size=stimuli_size)
        image = self._center_on_background(image, background_size=pixels_central_vision)
        image_path = self._write(image, directory=output_directory, extension=os.path.splitext(image_path)[-1])
        image.close()
        return image_path

    def _load_image(self, image_path):
        return Image.open(image_path)

    def _resize_image(self, image, image_size):
        return image.resize(image_size, Image.ANTIALIAS)

    def _center_on_background(self, center_image, background_size, background_color='gray'):
        image = Image.new('RGB', background_size, background_color)
        center_topleft = (np.subtract(background_size, center_image.size) / 2).round().astype(int)
        image.paste(center_image, tuple(center_topleft))
        return image

    def _write(self, image, directory, extension=None):
        image_hash = hashlib.sha1(image.tobytes()).hexdigest()
        output_path = os.path.join(directory, image_hash) + (extension or '')
        image.save(output_path)
        return output_path


class AttributeModel(peewee.Model):
    name = peewee.CharField(unique=True)
    type = peewee.CharField()

    class Meta:
        database = pwdb


class ImageModel(peewee.Model):
    image_id = peewee.CharField()

    class Meta:
        database = pwdb


class ImageMetaModel(peewee.Model):
    image = peewee.ForeignKeyField(ImageModel, backref="image_meta_models")
    attribute = peewee.ForeignKeyField(AttributeModel, backref="image_meta_models")
    value = peewee.CharField()

    class Meta:
        database = pwdb


class StimulusSetModel(peewee.Model):
    name = peewee.CharField()

    class Meta:
        database = pwdb


class ImageStoreModel(peewee.Model):
    store_type = peewee.CharField()
    location = peewee.CharField()
    location_type = peewee.CharField()
    unique_name = peewee.CharField(unique=True, null=True, index=True)
    sha1 = peewee.CharField(unique=True, null=True, index=True)

    class Meta:
        database = pwdb


class StimulusSetImageMap(peewee.Model):
    stimulus_set = peewee.ForeignKeyField(StimulusSetModel, backref="stimulus_set_image_maps")
    image = peewee.ForeignKeyField(ImageModel, backref="stimulus_set_image_maps")

    class Meta:
        database = pwdb


class ImageStoreMap(peewee.Model):
    image_store = peewee.ForeignKeyField(ImageStoreModel, backref="image_image_store_maps")
    image = peewee.ForeignKeyField(ImageModel, backref="image_image_store_maps")
    path = peewee.CharField()

    class Meta:
        database = pwdb
