from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import logging
import os
import zipfile

import boto3
import pandas as pd
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from six.moves.urllib.parse import urlparse
from tqdm import tqdm

from brainscore import assemblies
from brainscore.assemblies import coords_for_dim
from brainscore.stimuli import StimulusSet, ImageModel, AttributeModel, ImageMetaModel, StimulusSetModel, \
    ImageStoreModel, StimulusSetImageMap, ImageStoreMap
from brainscore.utils import fullname

_local_data_path = os.path.expanduser("~/.brainscore/data")


class Fetcher(object):
    """A Fetcher obtains data with which to populate a DataAssembly.  """

    def __init__(self, location, unique_name):
        self.location = location
        self.unique_name = unique_name
        self.local_dir_path = os.path.join(_local_data_path, self.unique_name)
        os.makedirs(self.local_dir_path, exist_ok=True)

    def fetch(self):
        """
        Fetches the resource identified by location.
        :return: a full local file path
        """
        raise NotImplementedError("The base Fetcher class does not implement .fetch().  Use a subclass of Fetcher.  ")


class BotoFetcher(Fetcher):
    """A Fetcher that retrieves files from Amazon Web Services' S3 data storage.  """

    def __init__(self, location, unique_name):
        super(BotoFetcher, self).__init__(location, unique_name)
        self.parsed_url = urlparse(self.location)
        self.split_hostname = self.parsed_url.hostname.split(".")
        self.split_path = self.parsed_url.path.lstrip('/').split("/")
        # http://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html#access-bucket-intro
        virtual_hosted_style = len(self.split_hostname) == 4
        if virtual_hosted_style:
            self.bucketname = self.split_hostname[0]
            self.relative_path = os.path.join(*(self.split_path))
        else:
            self.bucketname = self.split_path[0]
            self.relative_path = os.path.join(*(self.split_path[1:]))
        self.basename = self.split_path[-1]
        self.output_filename = os.path.join(self.local_dir_path, self.relative_path)
        self._logger = logging.getLogger(fullname(self))

    def fetch(self):
        if not os.path.exists(self.output_filename):
            self.download_boto()
        return self.output_filename

    def download_boto(self, sha1=None):
        """Downloads file from S3 via boto at `url` and writes it in `self.output_filename`."""
        self._logger.info('downloading %s' % self.relative_path)
        try:  # try with authentication
            self._logger.debug("attempting default download (signed)")
            self.download_boto_config(config=None, sha1=sha1)
        except Exception:  # try without authentication
            self._logger.debug("default download failed, trying unsigned")
            # disable signing requests. see https://stackoverflow.com/a/34866092/2225200
            unsigned_config = Config(signature_version=UNSIGNED)
            self.download_boto_config(config=unsigned_config, sha1=sha1)

    def download_boto_config(self, config, sha1=None):
        s3 = boto3.resource('s3', config=config)
        obj = s3.Object(self.bucketname, self.relative_path)
        # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
        with tqdm(total=obj.content_length, unit='B', unit_scale=True, desc=self.relative_path) as progress_bar:
            def progress_hook(bytes_amount):
                progress_bar.update(bytes_amount)

            obj.download_file(self.output_filename, Callback=progress_hook)

        if sha1 is not None:
            self.verify_sha1(self.output_filename, sha1)

    def verify_sha1(self, filename, sha1):
        data = open(filename, 'rb').read()
        if sha1 != hashlib.sha1(data).hexdigest():
            raise IOError("File '%s': invalid SHA-1 hash! You may want to delete "
                          "this corrupted file..." % filename)


class URLFetcher(Fetcher):
    """A Fetcher that retrieves a resource identified by URL.  """

    def __init__(self, location, unique_name):
        super(URLFetcher, self).__init__(location, unique_name)


class LocalFetcher(Fetcher):
    """A Fetcher that retrieves local files.  """

    def __init__(self, location, unique_name):
        super(LocalFetcher, self).__init__(location, unique_name)


class AssemblyLoader(object):
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
        concatenated = xr.concat(data_arrays, dim="presentation")
        stimulus_set_name = self.assy_model.stimulus_set.name
        stimulus_set = get_stimulus_set(stimulus_set_name)
        merged = self.merge(concatenated, stimulus_set)
        class_object = getattr(assemblies, self.assy_model.assembly_class)
        result = class_object(data=merged)
        result.attrs["stimulus_set_name"] = stimulus_set_name
        result.attrs["stimulus_set"] = stimulus_set
        return result

    def merge(self, assy, stimulus_set):
        axis_name = "presentation"
        df_of_coords = pd.DataFrame(coords_for_dim(assy, axis_name))
        merged = df_of_coords.merge(stimulus_set, on="image_id", how="left")
        for col in stimulus_set.columns:
            assy[col] = (axis_name, merged[col])
            # assy.set_index(append=True, inplace=True, **{axis_name: [col]})
        return assy


class AssemblyFetchError(Exception):
    pass


_fetcher_types = {
    "local": LocalFetcher,
    "S3": BotoFetcher,
    "URL": URLFetcher
}


def get_fetcher(type="S3", location=None, unique_name=None):
    return _fetcher_types[type](location, unique_name)


def fetch_assembly(assy_model):
    local_paths = {}
    for s in assy_model.assembly_store_maps:
        fetcher = get_fetcher(type=s.assembly_store_model.location_type,
                              location=s.assembly_store_model.location,
                              unique_name=s.assembly_store_model.unique_name)
        local_paths[s.role] = fetcher.fetch()
    return local_paths


def fetch_stimulus_set(stimulus_set_model):
    local_paths = {}
    image_paths = {}
    pw_query_stores_for_set = ImageStoreModel.select() \
        .join(ImageStoreMap) \
        .join(ImageModel) \
        .join(StimulusSetImageMap) \
        .join(StimulusSetModel) \
        .where(StimulusSetModel.name == stimulus_set_model.name) \
        .distinct()
    for s in pw_query_stores_for_set:
        fetcher = get_fetcher(type=s.location_type,
                              location=s.location,
                              unique_name=s.unique_name)
        fetched = fetcher.fetch()
        containing_dir = os.path.dirname(fetched)
        with zipfile.ZipFile(fetched, 'r') as zip_file:
            if not all(map(lambda x: os.path.exists(os.path.join(containing_dir, x)), zip_file.namelist())):
                zip_file.extractall(containing_dir)
        local_paths[s.location] = containing_dir
    for image_map in stimulus_set_model.stimulus_set_image_maps.prefetch(ImageModel, ImageStoreMap, ImageStoreModel):
        store_map = image_map.image.image_image_store_maps[0]
        local_path_base = local_paths[store_map.image_store.location]
        image_path = os.path.join(local_path_base, store_map.path)
        image_paths[image_map.image.image_id] = image_path
    return image_paths


def get_assembly(name):
    assy_model = assemblies.lookup_assembly(name)
    local_paths = fetch_assembly(assy_model)
    loader = AssemblyLoader(assy_model, local_paths)
    assembly = loader.load()
    assembly.name = name
    return assembly


def get_stimulus_set(name):
    stimulus_set_model = StimulusSetModel.get(StimulusSetModel.name == name)
    image_paths = fetch_stimulus_set(stimulus_set_model)
    pw_query = ImageModel.select() \
        .join(StimulusSetImageMap) \
        .join(StimulusSetModel) \
        .where(StimulusSetModel.name == name)
    df_reconstructed = pd.DataFrame(list(pw_query.dicts()))
    pw_query_attributes = AttributeModel.select() \
        .join(ImageMetaModel) \
        .join(ImageModel) \
        .join(StimulusSetImageMap) \
        .join(StimulusSetModel) \
        .where(StimulusSetModel.name == name) \
        .distinct()
    for a in pw_query_attributes:
        pw_query_single_attribute = AttributeModel.select(ImageModel.image_id, ImageMetaModel.value) \
            .join(ImageMetaModel) \
            .join(ImageModel) \
            .join(StimulusSetImageMap) \
            .join(StimulusSetModel) \
            .where((StimulusSetModel.name == name) & (AttributeModel.name == a.name))
        df_single_attribute = pd.DataFrame(list(pw_query_single_attribute.dicts()))
        merged = df_reconstructed.merge(df_single_attribute, on="image_id", how="left", suffixes=("orig_", ""))
        df_reconstructed[a.name] = merged["value"].astype(a.type)
    stimulus_set = StimulusSet(df_reconstructed)
    stimulus_set.image_paths = image_paths
    return stimulus_set
