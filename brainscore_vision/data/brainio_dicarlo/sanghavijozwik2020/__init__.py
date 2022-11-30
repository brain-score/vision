import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

data_registry['dicarlo.SanghaviJozwik2020'] = lambda: load_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0")
