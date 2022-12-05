import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version id and sha1
data_registry['schiller.Schiller1976c'] = lambda: load_from_s3(
    identifier="schiller.Schiller1976c",
    version_id="",
    sha1="")
