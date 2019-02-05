import logging
import sys

import pytest

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for logger in ['peewee', 's3transfer', 'botocore', 'boto3', 'urllib3', 'PIL']:
    logging.getLogger(logger).setLevel(logging.INFO)

try:
    _SKIP_PRIVATE = not pytest.config.getoption("--skip-private")
except (ValueError, AttributeError):
    # Can't get config from pytest, e.g., because framework is installed instead
    # of being run from a development version (and hence conftests.py is not
    # available). Don't run private tests.
    _SKIP_PRIVATE = True

private_access = pytest.mark.skipif(
    _SKIP_PRIVATE, reason="set --skip-private option to not run tests that require private s3 access")
