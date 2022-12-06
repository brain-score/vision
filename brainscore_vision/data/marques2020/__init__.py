import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version id and sha1
data_registry['movshon.Cavanaugh2002a'] = lambda: load_from_s3(
    identifier="movshon.Cavanaugh2002a",
    version_id="",
    sha1="")

data_registry['movshon.FreemanZiemba2013_V1_properties'] = lambda: load_from_s3(
    identifier="'movshon.FreemanZiemba2013_V1_properties'",
    version_id="",
    sha1="")

data_registry['devalois.DeValois1982a'] = lambda: load_from_s3(
    identifier="devalois.DeValois1982a",
    version_id="",
    sha1="")

data_registry['devalois.DeValois1982b'] = lambda: load_from_s3(
    identifier="devalois.DeValois1982b",
    version_id="",
    sha1="")

data_registry['shapley.Ringach2002'] = lambda: load_from_s3(
    identifier="shapley.Ringach2002",
    version_id="",
    sha1="")

data_registry['schiller.Schiller1976c'] = lambda: load_from_s3(
    identifier="schiller.Schiller1976c",
    version_id="",
    sha1="")
