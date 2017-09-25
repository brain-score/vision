from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import sqlite3
from urllib.parse import urlparse

import boto as boto


class Fetcher(object):
    """A Fetcher obtains data with which to populate a DataAssembly.  """
    def __init__(self):
        pass


class BotoFetcher(Fetcher):
    """A Fetcher that retrieves files from Amazon Web Services' S3 data storage.  """
    def __init__(self):
        pass


class LocalFetcher(Fetcher):
    """A Fetcher that retrieves local files.  """
    def __init__(self):
        pass


class Lookup(object):
    """A Lookup gives an abstraction layer for findoing out where a DataAssembly can be fetched from.  """
    def __init__(self):
        pass


class SQLiteLookup(Lookup):
    """A Lookup that uses a local SQLite file.  """
    def __init__(self, db_file="lookup.db"):
        self.db_file = db_file

    def get_connection(self):
        if not hasattr(self, '_connection'):
            self._connection = sqlite3.connect(self.db_file)
        return self._connection

    def lookup_assembly(self, name):
        pass


class AssemblyRecord(object):
    """An AssemblyRecord stores information about the canonical location where the data for a DataAssembly is stored.  """
    def __init__(self, db_id, name, stores):
        self.db_id = db_id
        self.name = name
        self.stores = stores


class AssemblyStoreMap(object):
    """An AssemblyStoreMap links an AssemblyRecord to a Store.  """
    def __init__(self, db_id, role, store, assembly_record):
        self.db_id = db_id
        self.role = role
        self.store = store
        self.assembly_record = assembly_record


class Store(object):
    """A Store stores the location of a DataAssembly data file.  """
    def __init__(self, db_id, type, location):
        self.db_id = db_id
        self.type = type
        self.location = location


def download_boto(url, credentials=(None,), output_filename=None, sha1=None):
    """Downloads file from S3 via boto at `url` and writes it in `output_dirname`.
    Borrowed from dldata.  """

    boto_conn = boto.connect_s3(*credentials)
    url = urlparse(url)
    bucketname = url.netloc.split('.')[0]
    file = url.path.strip('/')
    bucket = boto_conn.get_bucket(bucketname)
    key = bucket.get_key(file)
    print('getting %s' % file)
    key.get_contents_to_filename(output_filename)

    if sha1 is not None:
        verify_sha1(output_filename, sha1)


def verify_sha1(filename, sha1):
    data = open(filename, 'rb').read()
    if sha1 != hashlib.sha1(data).hexdigest():
        raise IOError("File '%s': invalid SHA-1 hash! You may want to delete "
                      "this corrupted file..." % filename)

