from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import peewee

from mkgu.lookup import pwdb


class StimulusSet(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ["image_paths", "get_image"]
    _internal_names_set = set(_internal_names)

    def get_image(self, image_id):
        return self.image_paths[image_id]


class ImageModel(peewee.Model):
    hash_id = peewee.CharField()
    object_name = peewee.CharField()
    category_name = peewee.CharField()
    background_id = peewee.CharField()
    image_file_name = peewee.CharField()
    variation = peewee.IntegerField()
    ty = peewee.FloatField()
    tz = peewee.FloatField()
    rxy = peewee.FloatField()
    rxz = peewee.FloatField()
    ryz = peewee.FloatField()
    rxy_semantic = peewee.FloatField()
    rxz_semantic = peewee.FloatField()
    ryz_semantic = peewee.FloatField()
    size = peewee.FloatField()
    s = peewee.FloatField()

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


