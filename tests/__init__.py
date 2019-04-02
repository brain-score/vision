import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
for logger in ['peewee', 's3transfer', 'botocore', 'boto3', 'urllib3', 'PIL']:
    logging.getLogger(logger).setLevel(logging.INFO)
