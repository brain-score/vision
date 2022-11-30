import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version id and sha1
data_registry['devalois.DeValois1982a'] = lambda: load_from_s3(
    identifier="devalois.DeValois1982a",
    version_id="",
    sha1="")
