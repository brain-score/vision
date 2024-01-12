import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
for disable_logger in ['s3transfer', 'botocore', 'boto3', 'urllib3', 'peewee', 'PIL']:
    logging.getLogger(disable_logger).setLevel(logging.WARNING)
