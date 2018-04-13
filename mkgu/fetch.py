from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import os
import peewee
import boto3
import xarray as xr
from six.moves.urllib.parse import urlparse

from mkgu import assemblies
from mkgu.lookup import get_lookup

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


class Loader(object):
    """
    Loads a DataAssembly from files.
    """
    def __init__(self, assy_model, local_paths):
        self.assy_model = assy_model
        self.local_paths = local_paths

    def load(self):
        data_arrays = []
        for role, path in self.local_paths.items():
            tmp_da = xr.open_dataarray(path)
            data_arrays.append(tmp_da)
        merged = xr.concat(data_arrays, dim="presentation")
        class_object = getattr(assemblies, self.assy_model.assembly_class)
        result = class_object(data=merged)
        return result


class AssemblyFetchError(Exception):
    pass


_fetcher_types = {
    "local": LocalFetcher,
    "S3": BotoFetcher,
    "URL": URLFetcher
}


def get_fetcher(type="S3", location=None, assembly_name=None):
    return _fetcher_types[type](location, assembly_name)


def fetch_assembly(assy_model):
    local_paths = {}
    for s in assy_model.assembly_store_maps:
        fetcher = get_fetcher(type=s.assembly_store_model.location_type,
                              location=s.assembly_store_model.location,
                              assembly_name=assy_model.name)
        local_paths[s.role] = fetcher.fetch()
    return local_paths


def get_assembly(name):
    assy_model = get_lookup().lookup_assembly(name)
    local_paths = fetch_assembly(assy_model)
    loader = Loader(assy_model, local_paths)
    return loader.load()


