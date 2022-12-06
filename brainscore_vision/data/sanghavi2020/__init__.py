import logging

from brainscore_vision import data_registry
from brainscore_vision.utils.s3 import load_from_s3

_logger = logging.getLogger(__name__)

BIBTEX = """"""

data_registry['dicarlo.Sanghavi2020'] = lambda: load_from_s3(
    identifier="dicarlo.Sanghavi2020",
    version_id="",
    sha1="12e94e9dcda797c851021dfe818b64615c785866")

data_registry['dicarlo.SanghaviJozwik2020'] = lambda: load_from_s3(
    identifier="dicarlo.SanghaviJozwik2020",
    version_id="",
    sha1="c5841f1e7d2cf0544a6ee010e56e4e2eb0994ee0")

data_registry['dicarlo.SanghaviMurty2020'] = lambda: load_from_s3(
    identifier="dicarlo.SanghaviMurty2020",
    version_id="",
    sha1="6cb8e054688066d1d86d4944e1385efc6a69ebd4")