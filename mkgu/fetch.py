from __future__ import absolute_import, division, print_function, unicode_literals

import os
import hashlib
import sqlite3

import boto3
import xarray as xr
from six.moves.urllib.parse import urlparse
import requests

from mkgu import assemblies

_local_data_path = os.path.expanduser("~/.mkgu/data")


class Fetcher(object):
    """A Fetcher obtains data with which to populate a DataAssembly.  """
    def __init__(self, location, assembly_name):
        self.location = location
        self.assembly_name = assembly_name
        self.local_dir_path = os.path.join(_local_data_path, self.assembly_name)
        os.makedirs(self.local_dir_path, exist_ok=True)

    def fetch(self):
        """
        Fetches the resource identified by location.
        :return: a full local file path
        """
        raise NotImplementedError("The base Fetcher class does not implement .fetch().  Use a subclass of Fetcher.  ")


class BotoFetcher(Fetcher):
    """A Fetcher that retrieves files from Amazon Web Services' S3 data storage.  """
    def __init__(self, location, assembly_name):
        super(BotoFetcher, self).__init__(location, assembly_name)
        self.parsed_url = urlparse(self.location)
        self.split_hostname = self.parsed_url.hostname.split(".")
        self.split_path = self.parsed_url.path.lstrip('/').split("/")
        virtual_hosted_style = len(self.split_hostname) == 4 # http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#access-bucket-intro
        if virtual_hosted_style:
            self.bucketname = self.split_hostname[0]
            self.relative_path = os.path.join(*(self.split_path))
        else:
            self.bucketname = self.split_path[0]
            self.relative_path = os.path.join(*(self.split_path[1:]))
        self.basename = self.split_path[-1]
        self.output_filename = os.path.join(self.local_dir_path, self.relative_path)

    def fetch(self):
        if not os.path.exists(self.output_filename):
            self.download_boto()
        return self.output_filename

    def download_boto(self, credentials=(None,), sha1=None):
        """Downloads file from S3 via boto at `url` and writes it in `output_dirname`.
        Borrowed from dldata.  """
        s3 = boto3.client('s3')
        print('getting %s' % self.relative_path)
        s3.download_file(self.bucketname, self.relative_path, self.output_filename)

        if sha1 is not None:
            self.verify_sha1(self.output_filename, sha1)

    def verify_sha1(self, filename, sha1):
        data = open(filename, 'rb').read()
        if sha1 != hashlib.sha1(data).hexdigest():
            raise IOError("File '%s': invalid SHA-1 hash! You may want to delete "
                          "this corrupted file..." % filename)


class URLFetcher(Fetcher):
    """A Fetcher that retrieves a resource identified by URL.  """
    def __init__(self, location, assembly_name):
        super(URLFetcher, self).__init__(location, assembly_name)


class LocalFetcher(Fetcher):
    """A Fetcher that retrieves local files.  """
    def __init__(self, location, assembly_name):
        super(LocalFetcher, self).__init__(location, assembly_name)


class Lookup(object):
    """A Lookup gives an abstraction layer for findoing out where a DataAssembly can be fetched from.  """
    def __init__(self):
        pass

    def lookup_assembly(self, name):
        pass


class SQLiteLookup(Lookup):
    """A Lookup that uses a local SQLite file.  """
    sql_lookup_assy = """SELECT
    a.id as a_id, a.name, a.class
    FROM
    assembly a
    WHERE
    a.name = ?
    """

    sql_get_assy = """SELECT
    a.id as a_id, a.name, a.class, a_s.id as a_s_id, a_s.role,
    s.id as s_id, s.type, s.location
    FROM
    assembly a
    JOIN assembly_store a_s ON a.id = a_s.assembly_id
    JOIN store s ON a_s.store_id = s.id
    WHERE
    a.name = ?
    """

    def __init__(self, db_file=None):
        super(SQLiteLookup, self).__init__()
        if db_file is None:
            db_file = os.path.join(os.path.dirname(__file__), "lookup.db")
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
        if assy_result:
            assy = AssemblyRecord(assy_result["a_id"], assy_result["name"], assy_result["class"])
            cursor.execute(self.sql_get_assy, (name,))
            assy_store_result = cursor.fetchall()
            for r in assy_store_result:
                s = Store(r["s_id"], r["type"], r["location"], [assy])
                role = r["role"]
                a_s = AssemblyStoreMap(r["a_s_id"], role, s, assy)
                assy.stores[role] = a_s
            return assy
        else:
            raise AssemblyLookupError("A DataAssembly named " + name + " was not found.")


class PostgreSQLLookup(Lookup):
    """A Lookup that uses a Postgres database.  """
    def __init__(self):
        super(PostgreSQLLookup, self).__init__()


class WebServiceLookup(Lookup):
    """A Lookup that uses a web service.  """
    def __init__(self):
        super(WebServiceLookup, self).__init__()


class AssemblyRecord(object):
    """An AssemblyRecord stores information about the canonical location where the data
    for a DataAssembly is stored.  """
    def __init__(self, db_id, name, cls, stores=None):
        self.db_id = db_id
        self.name = name
        self.cls = cls
        if stores is None:
            stores = {}
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
    def __init__(self, db_id, type, location, assemblies=None):
        self.db_id = db_id
        self.type = type
        self.location = location
        if assemblies is None:
            assemblies = []
        self.assemblies = assemblies


class Loader(object):
    """
    Loads a DataAssembly from files.
    """
    def __init__(self, assy_record, local_paths):
        self.assy_record = assy_record
        self.local_paths = local_paths

    def coords_for_dim(self, xr_data, dim):
        return [x[0] for x in xr_data.coords.variables.items() if x[1].dims == (dim,)]

    def gather_indexes(self, xr_data):
        xr_data.set_index(append=True, inplace=True, **{dim: self.coords_for_dim(xr_data, dim) for dim in xr_data.dims})
        return xr_data

    def load(self):
        data_arrays = []
        for role, path in self.local_paths.items():
            tmp_da = xr.open_dataarray(path)
            # tmp_da = Assembly(tmp_da)
            data_arrays.append(self.gather_indexes(tmp_da))
        merged = xr.concat(data_arrays, dim="presentation")
        class_object = getattr(assemblies, self.assy_record.cls)
        result = class_object(merged, name=self.assy_record.name)
        return result


class AssemblyLookupError(Exception):
    pass


class AssemblyFetchError(Exception):
    pass


_fetcher_types = {
    "local": LocalFetcher,
    "S3": BotoFetcher,
    "URL": URLFetcher
}


def get_fetcher(type="S3", location=None, assembly_name=None):
    return _fetcher_types[type](location, assembly_name)


_lookup_types = {
    "SQLite": SQLiteLookup,
    "PostgreSQL": PostgreSQLLookup,
    "WebService": WebServiceLookup
}


def get_lookup(type="SQLite"):
    return _lookup_types[type]()


def fetch_assembly(assy_record):
    local_paths = {}
    for s in assy_record.stores.values():
        fetcher = get_fetcher(type=s.store.type, location=s.store.location, assembly_name=assy_record.name)
        local_paths[s.role] = fetcher.fetch()
    return local_paths


def get_assembly(name):
    assy_record = get_lookup().lookup_assembly(name)
    local_paths = fetch_assembly(assy_record)
    loader = Loader(assy_record, local_paths)
    return loader.load()


