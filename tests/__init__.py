import logging
import sys

import pytest

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for logger in ['peewee', 's3transfer', 'botocore', 'boto3', 'urllib3', 'PIL']:
    logging.getLogger(logger).setLevel(logging.INFO)

try:
    _SKIP_PRIVATE = not pytest.config.getoption("--skip-private")
    _SKIP_GPU = not pytest.config.getoption("--skip-gpu")
    _SKIP_MEMORY = not pytest.config.getoption("--skip-memory-intense")
except (ValueError, AttributeError):
    # Can't get config from pytest, e.g., because framework is installed instead
    # of being run from a development version (and hence conftests.py is not
    # available). Don't run private tests.
    _SKIP_PRIVATE = True
    _SKIP_GPU = True
    _SKIP_MEMORY = True

private_access = pytest.mark.skipif(
    _SKIP_PRIVATE, reason="set to not run tests that require private s3 access")
requires_gpu = pytest.mark.skipif(
    _SKIP_GPU, reason="set to not run tests that require a GPU to run")
memory_intense = pytest.mark.skipif(
    _SKIP_MEMORY,
    reason="set --skip-memory-intense option to not run memory intense tests")
