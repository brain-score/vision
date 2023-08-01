"""
Point brainio to the local lookup file.
"""

import logging
from pathlib import Path

from brainio.catalogs import Catalog

_logger = logging.getLogger(__name__)


def brainio_brainscore():
    path = Path(__file__).parent / "lookup.csv"
    _logger.debug(f"Loading lookup from {path}")
    print(f"Loading lookup from {path}")  # print because logging usually isn't set up at this point during import
    catalog = Catalog.from_files("brainio_brainscore", path)  # setup.py is where the entrypoint's published name is set
    return catalog


