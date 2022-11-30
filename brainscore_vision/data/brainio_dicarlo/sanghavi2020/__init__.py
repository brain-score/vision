import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

data_registry['dicarlo.Sanghavi2020'] = lambda: load_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="",
    sha1="12e94e9dcda797c851021dfe818b64615c785866")
