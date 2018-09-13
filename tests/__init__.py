import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)-15s %(levelname)s:%(name)s:%(message)s')
logging.getLogger("peewee").setLevel(logging.WARNING)
