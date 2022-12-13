import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

# TODO: add correct version id and sha1
data_registry['dicarlo.Rajalingham2020'] = lambda: load_from_s3(
    identifier="dicarlo.Rajalingham2020",
    version_id="",
    sha1="ab95ae6c9907438f87b9b13b238244049f588680")

