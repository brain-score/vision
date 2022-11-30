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
