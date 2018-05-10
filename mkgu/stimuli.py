from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import peewee

from mkgu.lookup import pwdb


class StimulusSet(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ["image_paths", "get_image"]
    _internal_names_set = set(_internal_names)

    def get_image(self, image_id):
        return self.image_paths[image_id]


class AttributeModel(peewee.Model):
    name = peewee.CharField(unique=True)
    type = peewee.CharField()

    class Meta:
        database = pwdb


class ImageModel(peewee.Model):
    hash_id = peewee.CharField()

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


