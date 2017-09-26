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

    def lookup_assembly(self, name):
        pass


class SQLiteLookup(Lookup):
    """A Lookup that uses a local SQLite file.  """
    sql_lookup_assy = """SELECT 
    a.id as a_id, a.name 
    FROM
    assembly a
    WHERE
    a.name = ?
    """

    sql_get_assy = """SELECT 
    a.id as a_id, a.name, a_s.id as a_s_id, a_s.role, s.id as s_id, s.type, s.location 
    FROM
    assembly a
    JOIN assembly_store a_s ON a.id = a_s.assembly_id
    JOIN store s ON a_s.store_id = s.id
    WHERE
    a.name = ?
    """

    def __init__(self, db_file="lookup.db"):
        self.db_file = db_file

    def get_connection(self):
        if not hasattr(self, '_connection'):
            self._connection = sqlite3.connect(self.db_file)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def lookup_assembly(self, name):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(self.sql_lookup_assy, (name,))
        assy_result = cursor.fetchone()
        assy = AssemblyRecord(assy_result["a_id"], assy_result["name"], [])
        cursor.execute(self.sql_get_assy, (name,))
        assy_store_result = cursor.fetchall()
        for r in assy_store_result:
            s = Store(r["s_id"], r["type"], r["location"], [assy])
            role = r["role"]
            a_s = AssemblyStoreMap(r["a_s_id"], role, s, assy)
            assy.stores[role] = a_s
        return assy


lookup_types = {
    "SQLite": SQLiteLookup
}


def get_lookup(type="SQLite"):
    return lookup_types[type]()


class AssemblyRecord(object):
    """An AssemblyRecord stores information about the canonical location where the data
    for a DataAssembly is stored.  """
    def __init__(self, db_id, name, stores={}):
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
    def __init__(self, db_id, type, location, assemblies=[]):
        self.db_id = db_id
        self.type = type
        self.location = location
        self.assemblies = assemblies


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

